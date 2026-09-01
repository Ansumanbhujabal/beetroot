"""The single LLM abstraction.

Nobody writes an HTTP client for a model provider. Every model call — chat
completion or embedding — goes through `litellm.completion` /
`litellm.embedding` with a model string, `num_retries`, and a `fallbacks`
list. Cost comes from `litellm.completion_cost`, never a hand-maintained
price table that silently rots when a deployment changes. Switching provider
is a model string in `config/beatroot.yaml`, not a code change. Spec §14.
"""

import hashlib
import json
import logging
import re

import litellm
from pydantic import BaseModel

from beatroot.contracts.trust import Completion, CostRecord
from beatroot.obs.logging import current_request_id
from beatroot.obs.tracing import observe_generation, record_generation_result
from beatroot.reasoning.prompts import Prompt
from beatroot.settings import get_settings

log = logging.getLogger("beatroot.llm")
litellm.drop_params = True  # tolerate provider-specific param gaps
litellm.suppress_debug_info = True
# SOCKET LEAK. LiteLLM's aiohttp transport opens a `ClientSession` per call
# and never closes it: measured 3 calls -> 3 leaked sessions, each surfacing
# later as an `asyncio: Unclosed client session` / `Unclosed connector` ERROR
# when the GC finally finalises it. A single /recommend makes ~3 model calls,
# so a server leaks roughly one socket per call until it runs out of file
# descriptors — the errors are the symptom, the exhaustion is the bug.
#
# Routing through httpx instead, which pools and closes connections properly,
# takes the leak to zero (verified by counting live `aiohttp.ClientSession`
# objects before and after a batch of calls). Deliberately NOT fixed by
# filtering the log line: `obs.logging` already silences one *benign* LiteLLM
# teardown message and warns against widening that filter — silencing this
# one would have hidden a real resource leak behind a quieter log.
litellm.disable_aiohttp_transport = True
# LiteLLM's own logger emits INFO chatter on every call ("Wrapper: Completed
# Call...") which drowns the CLI's actual output and would drown a screen
# recording. Our structured logs already carry stage, cost and correlation id.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Offline stub only. Sign-hashing (see `_offline_vector`) keeps a bucket
# collision cheap rather than catastrophic, but the margin still narrows as
# the catalog vocabulary grows — at 256 dims, against a catalog vocabulary
# of roughly 256 distinct tokens, about two-thirds of tokens were already
# sharing a bucket. 1024 gives real headroom without materially changing
# cost (this vector never leaves the process or a local cache row).
EMBED_DIM = 1024

# Offline stub for `retrieval.query_rewrite` (stage="rewrite_query") only —
# see `LLMClient._offline_rewrite_query`. Small and hand-curated on purpose:
# this exists to make the rewrite step's effect visible and reproducible
# with no credentials, not to be a real synonym engine. A real provider
# replaces this entirely via `prompts/rewrite_query.md`; nothing else about
# `retrieval.query_rewrite.rewrite_query` changes.
_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "warm": ("hearty", "soup", "stew"),
    "comforting": ("hearty", "homestyle", "soulful"),
    "comfort": ("hearty", "homestyle"),
    "cold": ("chilled", "salad", "refreshing"),
    "cool": ("chilled", "refreshing"),
    "light": ("fresh", "salad", "low-calorie"),
    "heavy": ("rich", "filling"),
    "quick": ("fast", "easy", "simple"),
    "fast": ("quick", "easy"),
    "easy": ("simple", "quick"),
    "spicy": ("chili", "hot", "fiery"),
    "mild": ("gentle", "subtle"),
    "healthy": ("nutritious", "wholesome", "balanced"),
    "breakfast": ("morning", "brunch"),
    "lunch": ("midday",),
    "dinner": ("evening", "supper"),
    "sweet": ("dessert", "treat"),
    "rich": ("creamy", "indulgent"),
    "protein": ("high-protein", "filling"),
    "rice": ("grain", "pilaf"),
    "soup": ("broth", "stew"),
}


_CODE_FENCE_RE = re.compile(r"```[ \t]*[a-zA-Z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _strip_code_fence(text: str) -> str | None:
    """Pull the content out of the first ```markdown code fence``` in
    `text`, if any. Returns `None` when there is no fence at all, so the
    caller can move on to its next extraction strategy."""
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else None


def _first_balanced_json_object(text: str) -> str | None:
    """Slice out the first balanced curly-brace object in `text`,
    respecting JSON string quoting so a closing brace inside a quoted
    string value doesn't end the span early. Last-resort extraction for a
    reply that mixes prose with JSON but isn't wrapped in a clean code
    fence. Returns `None` when no such span exists."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _parse_llm_json(text: str) -> dict[str, object] | None:
    """Extract a JSON object from a real model's chat reply.

    Why this exists: unlike the offline stub, which always returns clean
    JSON, a real model routinely wraps its JSON reply in a markdown code
    fence, optionally tagged json, and sometimes adds a sentence of prose
    before or after it. A plain `json.loads` call chokes on the fence and
    raises, which used to make every JSON-dependent feature (rerank, query
    rewrite, self-assessment capture) silently degrade to its no-op
    fallback against any real provider, while the whole test suite,
    backed by a stub that never fences anything, stayed green. This is
    the works-offline, broken-live gap; see this module's own docstring.

    Tries, in order: plain `json.loads` on the raw text, then on the
    fence-stripped text, then on the first balanced curly-brace object
    found anywhere in the text — each only kept if it decodes to a JSON
    object (a bare array or scalar is not the shape a schema-bound caller
    wants). Returns `None`, never raises, when nothing in the text parses
    as a JSON object.
    """
    for candidate in (text, _strip_code_fence(text), _first_balanced_json_object(text)):
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class LLMClient:
    """The single LLM abstraction. Routing, retries, fallbacks, timeouts and
    cost accounting are LiteLLM's job, not ours. Switching provider is a model
    string in config/beatroot.yaml. Spec §14."""

    def __init__(
        self,
        model: str | None = None,
        fallbacks: list[str] | None = None,
        embedding_model: str | None = None,
        embedding_fallbacks: list[str] | None = None,
        offline: bool = False,
    ) -> None:
        cfg = get_settings().llm
        self.model = model or cfg.model
        self.fallbacks = fallbacks if fallbacks is not None else cfg.fallbacks
        self.embedding_model = embedding_model or cfg.embedding_model
        self.embedding_fallbacks = (
            embedding_fallbacks if embedding_fallbacks is not None else cfg.embedding_fallbacks
        )
        self.cfg = cfg
        self._offline = offline

    @classmethod
    def offline(cls) -> "LLMClient":
        """Deterministic, network-free client so the whole suite and a fresh
        clone run with no credentials."""
        return cls(offline=True)

    @property
    def embedder_id(self) -> str:
        """Identity of whatever `.embed()` actually calls. `"echo"` when
        offline — the stub hashes tokens, ignoring `embedding_model`
        entirely, so every offline run shares one vector space regardless
        of which model string is configured — otherwise the concrete
        `embedding_model`. `store.cache.EmbeddingCache` keys on this
        alongside the text so a real model's vectors and the offline
        stub's vectors (or two different real models') can never collide
        in the cache — they are not interchangeable vector spaces."""
        return "echo" if (self._offline or self._local_embeddings) else self.embedding_model

    @staticmethod
    def _trace_metadata(stage: str, prompt_ref: Prompt | None) -> dict[str, object]:
        """Metadata LiteLLM forwards to the Langfuse OTel exporter.

        Only the keys `litellm.integrations.langfuse.langfuse_otel` actually
        maps to span attributes are used — `session_id`, `trace_name`,
        `tags`, `trace_metadata` — because a key it does not recognise is
        carried nowhere and would be a comment pretending to be wiring.

        `session_id` is the request correlation id, which is what groups a
        single plan's THREE serial model calls (compile, rerank, explain)
        into one unit in Langfuse. Without it each generation appears
        unrelated and per-query cost has to be reassembled by hand from
        timestamps; with it, the cost of answering one user question is a
        number you can read straight off the session. Outside a request
        (CLI, eval runner) there is no correlation id and the key is simply
        omitted rather than filled with a placeholder that would group
        unrelated runs together.
        """
        metadata: dict[str, object] = {"stage": stage}
        tags = [f"stage:{stage}"] if stage else []
        if prompt_ref is not None:
            metadata["trace_metadata"] = prompt_ref.trace_metadata()
            tags.append(f"prompt:{prompt_ref.ref}")
        if stage:
            metadata["trace_name"] = f"beatroot.{stage}"
        if tags:
            metadata["tags"] = tags
        request_id = current_request_id()
        if request_id:
            metadata["session_id"] = request_id
        return metadata

    def complete(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        stage: str = "",
        prompt_ref: Prompt | None = None,
    ) -> Completion:
        """Send one completion. `prompt` is already-rendered text; `prompt_ref`
        is the `Prompt` object it was rendered FROM, carried purely so the
        trace can name the exact prompt version responsible for this
        generation. It is optional and never affects the request — a caller
        that omits it gets identical model behaviour and a slightly less
        informative trace."""
        if self._offline:
            return self._offline_completion(prompt, stage)

        with observe_generation(stage, prompt, self.model, prompt_ref) as generation:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.temperature,
                timeout=self.cfg.timeout_seconds,
                num_retries=self.cfg.max_retries,
                fallbacks=self.fallbacks,
                response_format={"type": "json_object"} if schema else None,
                metadata=self._trace_metadata(stage, prompt_ref),
            )
            text = response.choices[0].message.content or ""

            # A model returning prose where JSON was demanded is an ordinary
            # Tuesday, not a crash: always attempt the parse (callers may read
            # `self_assessment` off any completion, not only schema-bound ones),
            # and fall back to `parsed=None` with `text` preserved on failure.
            parsed = _parse_llm_json(text)
            if parsed is None and schema is not None:
                # Only a stage that actually asked for JSON (passed `schema=`)
                # gets this warning. `explain` (and any other prose-by-design
                # stage) never passes `schema` and returning prose there is
                # correct behaviour, not a parse failure worth flagging — see
                # every `complete()` call site for which stages pass `schema`.
                log.warning("stage=%s returned non-JSON; treating as unparsed", stage)

            try:
                usd = float(litellm.completion_cost(completion_response=response))
            except Exception as exc:
                log.warning("stage=%s completion_cost failed (%s); using 0.0", stage, exc)
                usd = 0.0

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            # Report the cost and token counts we ALREADY computed rather
            # than letting the tracing backend re-derive them from a model
            # name it may price differently — `/metrics` and the Langfuse
            # trace then agree by construction, not by coincidence. This
            # must happen INSIDE the `with`: leaving the block ends the
            # observation, and an update after that point is writing to a
            # span that has already been handed to the exporter.
            record_generation_result(
                generation,
                output=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usd=usd,
            )

        return Completion(
            text=text,
            parsed=parsed,
            self_assessment=(parsed or {}).get("self_assessment"),
            cost=CostRecord(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usd=usd,
                per_stage={stage: usd} if stage else {},
            ),
        )

    @property
    def _local_embeddings(self) -> bool:
        """True when embeddings are deliberately local while chat may still be
        a real hosted model — `embedding_model="local"`. Distinct from
        `_offline`, which makes *everything* deterministic."""
        return self.embedding_model == "local"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._offline or self._local_embeddings:
            return [self._offline_vector(t) for t in texts]
        response = litellm.embedding(
            model=self.embedding_model,
            input=texts,
            timeout=self.cfg.timeout_seconds,
            num_retries=self.cfg.max_retries,
            fallbacks=self.embedding_fallbacks,
        )
        return [d["embedding"] for d in response.data]

    # ---- offline stubs -------------------------------------------------
    def _offline_completion(self, prompt: str, stage: str) -> Completion:
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        parsed: dict[str, object] = {
            "choice_index": 0,
            "rationale": "offline deterministic choice",
            "self_assessment": 0.5,
        }
        if stage == "compile":
            # Deterministic, still-no-network parse for `compile_node`
            # (agent.nodes): every known catalog tag that appears, as a
            # whole word, in the free text becomes an `exclude_tags`
            # entry. Without this the fixed dict above (no `exclude_tags`
            # key at all) meant `compile_node`'s parse/filter logic never
            # ran on any credential-free path — `--preferences` visibly
            # did nothing offline, and golden case g30 passed without
            # exercising it. Reads both the known-tag list and the free
            # text straight off the rendered prompt (`compile_node` embeds
            # both), so this needs no catalog access of its own.
            parsed["exclude_tags"] = self._offline_extract_known_tags(prompt)
            parsed["prefer_tags"] = []
        elif stage == "rewrite_query":
            # Deterministic, still-no-network expansion for `retrieval.
            # query_rewrite.rewrite_query`: without this, the offline stub's
            # generic dict (no `rewritten_query` key) would make the rewrite
            # step degrade to a no-op on every credential-free run, and the
            # "something warm and comforting" demo would never show an
            # effect offline. See `_offline_rewrite_query`.
            parsed.update(self._offline_rewrite_query(prompt))
        text = f"[offline:{digest[:8]}]"
        if stage == "explain":
            text = self._offline_explain(prompt, digest)
        return Completion(
            text=text,
            parsed=parsed,
            self_assessment=0.5,
            cost=CostRecord(per_stage={stage: 0.0} if stage else {}),
        )

    @staticmethod
    def _offline_explain(prompt: str, digest: str) -> str:
        """Deterministic offline prose that actually STATES NUMBERS.

        This closes the A6 gap. The generic stub text (`[offline:a1b2c3d4]`)
        contains no digits, so `verify_node`'s `detect_drift` had nothing to
        parse and A6 (`explanation_grounding`) scored 1.000 on every offline
        run without ever exercising the check — the axis could not fail, so
        its passing meant nothing. `EVAL_RESULTS.md` and the readiness audit
        both disclosed this; it was never fixed because the stub is the only
        thing that runs in CI.

        The numbers come off the prompt's own "Verified facts" block, which
        `prompts/explain.md` renders as `kcal=520.0, protein_g=28.0, ...`.
        Same technique as `_offline_extract_known_tags`: read what the caller
        already embedded rather than reaching for the catalog.

        Grounded BY CONSTRUCTION, and that is the point — a stub stating the
        true numbers makes A6 measure the real path end to end (prose ->
        regex -> ledger -> COMMIT) instead of measuring silence. That it
        cannot drift on its own is why
        `tests/eval/test_a6_is_falsifiable.py` mutates this function to lie
        and asserts the system catches it; a check that has never rejected
        anything is not yet evidence of anything.
        """
        facts: dict[str, str] = {}
        for line in prompt.splitlines():
            if "=" in line and ("kcal=" in line or "protein_g=" in line):
                for part in line.split(","):
                    key, _, value = part.strip().partition("=")
                    if key and value:
                        facts[key.strip()] = value.strip()
                break
        if not facts:
            return f"[offline:{digest[:8]}]"
        kcal = facts.get("kcal", "?")
        protein = facts.get("protein_g", "?")
        sodium = facts.get("sodium_mg")
        parts = [f"This meal provides {kcal} kcal and {protein} g of protein"]
        if sodium:
            parts.append(f"with {sodium} mg of sodium")
        return ", ".join(parts) + ". Offline deterministic explanation."

    @staticmethod
    def _offline_extract_known_tags(prompt: str) -> list[str]:
        """Every tag on the prompt's own "Known tags:" line that also
        appears, as a whole word, in its "User text:" section — a simple
        plural ("peanuts" for the tag "peanut") and an underscore-as-space
        spelling ("tree nut" for the tag "tree_nut") both count as a
        mention. Deterministic and offline: no catalog access, no
        randomness, no network — just literal matching against text
        `compile_node` (`agent.nodes`) already rendered into `prompt` via
        `prompts/compile_constraints.md`.
        """
        known_match = re.search(r"Known tags:\s*(.*)", prompt)
        text_match = re.search(r"User text:\s*\n(.*?)\n\nReply", prompt, re.DOTALL)
        if not known_match or not text_match:
            return []
        known = [t.strip() for t in known_match.group(1).split(",") if t.strip()]
        free_text = text_match.group(1)
        found = []
        for tag in known:
            variants = {tag, tag.replace("_", " ")}
            if any(re.search(rf"\b{re.escape(v)}s?\b", free_text, re.IGNORECASE) for v in variants):
                found.append(tag)
        return found

    @staticmethod
    def _offline_rewrite_query(prompt: str) -> dict[str, object]:
        """Deterministic, still-no-network expansion for
        `retrieval.query_rewrite.rewrite_query`: every word in the query
        that appears in `_QUERY_EXPANSIONS` below contributes its fixed set
        of related terms, in table order, each only once. "something warm
        and comforting" -> "warm" and "comforting" each match, contributing
        hearty/soup/stew and homestyle/soulful — a real, reproducible
        expansion with no model and no randomness. A query with no matching
        word rewrites to itself (`terms == []`), which is a correct,
        honest answer, not a bug: not every query has something to expand.
        """
        match = re.search(r"User query:\s*\n(.*?)\n\nReply", prompt, re.DOTALL)
        query = match.group(1).strip() if match else ""
        tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]
        terms: list[str] = []
        for token in tokens:
            for extra in _QUERY_EXPANSIONS.get(token, ()):
                if extra not in terms:
                    terms.append(extra)
        rewritten = f"{query} {' '.join(terms)}".strip() if terms else query
        return {"rewritten_query": rewritten, "terms": terms}

    @staticmethod
    def _offline_vector(text: str) -> list[float]:
        """Deterministic bag-of-tokens embedding (the hashing trick).

        Hashing the WHOLE string gives near-identical text unrelated vectors,
        which makes dense retrieval anti-signal rather than merely weak. Hashing
        per TOKEN puts shared vocabulary in shared dimensions, so overlapping
        text scores as similar.

        This is not semantic — it will not connect "warm" to "hot". It rewards
        lexical overlap in a dense-vector shape, which is what an offline demo
        needs. A real embedding model replaces it via config; nothing else
        changes.
        """
        tokens = [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]
        if not tokens:
            # Preserve unit-norm for empty/unhashable input.
            return LLMClient._whole_string_vector(text)

        vec = [0.0] * EMBED_DIM
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % EMBED_DIM
            # Sign hashing: collisions cancel rather than compound.
            vec[index] += 1.0 if digest[4] & 1 else -1.0

        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else LLMClient._whole_string_vector(text)

    @staticmethod
    def _whole_string_vector(text: str) -> list[float]:
        """Fallback for text with no hashable tokens (empty string, pure
        punctuation) — still deterministic and unit-norm, just not
        token-sensitive since there are no tokens to be sensitive to."""
        seed = hashlib.sha256(text.encode()).digest()
        raw = [((seed[i % len(seed)] * (i + 7)) % 251) / 251.0 - 0.5 for i in range(EMBED_DIM)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]


def get_llm_client() -> LLMClient:
    """The single entry point every caller should use to obtain a client.
    Reads offline-mode through `get_settings()` — settings.py is the only
    module allowed to touch the process environment directly, so this stays
    a plain settings read rather than a direct environment lookup of its own."""
    return LLMClient.offline() if get_settings().offline else LLMClient()
