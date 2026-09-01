"""Group incidents into repeated patterns. Spec §9.

`propose` (see `heal.proposals`) only escalates a pattern into a
human-reviewable rule change once it has been SEEN repeatedly — a single
incident is noise, not a signal. This module is what decides "repeated":
incidents cluster together when they share a `kind` and a `detail` with the
numbers stripped out, so "kcal stated 780 vs computed 520" and "kcal stated
640 vs computed 430" are recognised as the same underlying drift pattern
even though no two incidents' text is byte-identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from beatroot.contracts.result import Incident

_NUMBERS = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Cluster:
    """One repeated pattern: same `kind`, same detail shape once digits are
    stripped. `incidents` is the full membership, in the order `cluster()`
    encountered them — never subsampled, so a caller can always recover
    every original `Incident` behind a proposal."""

    signature: str
    kind: str
    incidents: list[Incident] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.incidents)


def _signature(inc: Incident) -> str:
    """Stable grouping key: kind plus detail with every number replaced by
    `#`. Numbers are exactly the part of an incident's text that varies run
    to run (kcal counts, percentages, ids) — stripping them is what lets
    structurally identical failures cluster instead of each forming its own
    singleton."""
    return f"{inc.kind}:{_NUMBERS.sub('#', inc.detail).strip()}"


def cluster(incidents: list[Incident]) -> list[Cluster]:
    """Group `incidents` by `_signature`. Two incidents of different `kind`
    never land in the same cluster even if their stripped detail text
    happens to coincide — the signature is namespaced by kind precisely to
    prevent that.

    Returned in descending `count` order (ties broken by signature) so the
    caller — and a human reading `heal` output — sees the most-repeated,
    most-actionable patterns first.
    """
    buckets: dict[str, Cluster] = {}
    for inc in incidents:
        sig = _signature(inc)
        buckets.setdefault(sig, Cluster(signature=sig, kind=inc.kind)).incidents.append(inc)
    return sorted(buckets.values(), key=lambda c: (-c.count, c.signature))
