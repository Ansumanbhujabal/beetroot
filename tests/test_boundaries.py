"""The tier separation is a property of the code, not a claim in a README.

If a later task needs T0 to call an LLM, the design is wrong — fix the design,
not this test. Spec §3.

Note: importlib.import_module("beatroot.reasoning") is invisible to static
analysis, so a future reader should not assume total coverage.
"""

import ast
import pathlib
from collections import deque

import pytest

SRC = pathlib.Path(__file__).parents[1] / "src" / "beatroot"

FORBIDDEN = {
    "t0_invariants": {"beatroot.reasoning"},
    "trusted": {"beatroot.reasoning"},
    "store": {"beatroot.reasoning"},
}


def _module_name(path: pathlib.Path, root: pathlib.Path = SRC) -> str:
    """Dotted module name for a file under root/.

    An __init__.py IS its package: src/beatroot/trusted/__init__.py is
    `beatroot.trusted`, never `beatroot.trusted.__init__`. Getting this wrong
    de-links every package-level import from the graph.

    root: directory to use as base; defaults to module-level SRC.
    """
    parts = path.relative_to(root).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("beatroot", *parts))


def _package_of(path: pathlib.Path, root: pathlib.Path = SRC) -> str:
    """Containing package for relative-import resolution.

    A regular module's package is its parent (beatroot.trusted.catalog ->
    beatroot.trusted). An __init__.py IS its package, so it must NOT be
    stripped — stripping it resolves relative imports one level too high.

    root: directory to use as base for _module_name; defaults to module-level SRC.
    """
    module = _module_name(path, root)
    return module if path.name == "__init__.py" else module.rsplit(".", 1)[0]


def _imports(path: pathlib.Path, root: pathlib.Path = SRC) -> set[str]:
    """Extract all imported module names from a file, resolving relative imports.

    root: directory to use as base for package calculation; defaults to module-level SRC.
    """
    tree = ast.parse(path.read_text())
    package = _package_of(path, root)
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = package.split(".")
                # level 1 == current package, level 2 == parent, ...
                base = base[: len(base) - (node.level - 1)] or ["beatroot"]
                prefix = ".".join(base)
                module = f"{prefix}.{node.module}" if node.module else prefix
            else:
                module = node.module or ""
            found.add(module)
            # `from X import Y` may name a MODULE, not just a symbol
            found |= {f"{module}.{a.name}" for a in node.names if module}

    return found


def _modules(package: str) -> list[pathlib.Path]:
    return sorted((SRC / package).rglob("*.py"))


def _internal_graph(root: pathlib.Path = SRC) -> dict[str, set[str]]:
    """Build import graph of all beatroot modules under root.

    root: directory to scan; defaults to module-level SRC.

    Raises: ValueError naming both source files if a module name appears
    twice (e.g. a `trusted/` package directory and a sibling `trusted.py`
    both resolve to `beatroot.trusted` — the filesystem does not prevent
    this). `graph` maps module -> import set, so it cannot supply the first
    file's path for the error message; `sources` tracks that separately.
    """
    graph: dict[str, set[str]] = {}
    sources: dict[str, pathlib.Path] = {}
    for path in root.rglob("*.py"):
        module = _module_name(path, root)
        if module in sources:
            raise ValueError(
                f"Module collision: {module} appears twice ({sources[module]} and {path})"
            )
        sources[module] = path
        graph[module] = {i for i in _imports(path, root) if i.startswith("beatroot")}
    return graph


def _resolve_edge_to_node(edge: str, graph: dict[str, set[str]]) -> str | None:
    """Resolve an import edge name to a graph node key.

    An edge is either a graph key itself (if we imported a module), or it names
    a symbol inside a module whose key is the edge minus its last component
    (if we did `from X import Y`). Return the graph key or None if unresolved.

    Note: an edge resolving to bare "beatroot" almost always indicates a
    resolution bug upstream. In normal operation, such an edge should not exist.
    """
    if edge in graph:
        return edge
    # Try parent: from X import Y creates edge X.Y, but Y is in module X
    parent = edge.rsplit(".", 1)[0] if "." in edge else None
    if parent and parent in graph:
        return parent
    return None


def _reaches(start: str, target: str, graph: dict[str, set[str]]) -> list[str] | None:
    """BFS returning the offending path, so a failure names the whole chain."""
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, trail = queue.popleft()
        for nxt in graph.get(node, ()):
            if nxt == target or nxt.startswith(target + "."):
                return [*trail, nxt]
            # Resolve the edge name to a graph node key
            next_node = _resolve_edge_to_node(nxt, graph)
            if next_node and next_node not in seen:
                seen.add(next_node)
                queue.append((next_node, [*trail, next_node]))
    return None


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_tier_does_not_import_reasoning_directly(package):
    """Check that forbidden tiers do not directly import reasoning."""
    banned = FORBIDDEN[package]
    offenders = []
    for path in _modules(package):
        for imported in _imports(path):
            if any(imported == b or imported.startswith(b + ".") for b in banned):
                offenders.append(f"{path.relative_to(SRC)} imports {imported}")
    assert not offenders, (
        "Tier boundary violated — T0/trusted code must never reach the model:\n"
        + "\n".join(offenders)
    )


def test_no_tier_can_reach_reasoning_transitively():
    """Check that T0/trusted/store cannot reach reasoning through any import chain."""
    graph = _internal_graph()
    for package in FORBIDDEN:
        for path in _modules(package):
            module = _module_name(path)
            chain = _reaches(module, "beatroot.reasoning", graph)
            assert chain is None, (
                f"Tier boundary violated — {module} reaches the model through: "
                + " -> ".join(chain)
            )


def test_reachability_survives_a_package_init_hop(tmp_path):
    """Verify package __init__ hops work end-to-end via real files.

    If relative imports inside __init__.py break, `from . import X` produces
    edges one level too high (beatroot.X instead of beatroot.trusted.X), and
    package-level import chains silently vanish from the graph.
    """
    # Build a minimal tree: contracts/__init__.py imports reasoning,
    # t0_invariants/constraints.py imports beatroot.contracts
    root = tmp_path / "beatroot"
    root.mkdir()

    contracts = root / "contracts"
    contracts.mkdir()
    (contracts / "__init__.py").write_text("from beatroot import reasoning\n")

    t0 = root / "t0_invariants"
    t0.mkdir()
    (t0 / "__init__.py").write_text("")
    (t0 / "constraints.py").write_text("import beatroot.contracts\n")

    # Build graph from these files
    graph = _internal_graph(root)

    # The chain should be found: constraints → contracts → reasoning
    chain = _reaches("beatroot.t0_invariants.constraints", "beatroot.reasoning", graph)
    assert chain is not None, "package-level hop was not followed"
    assert "beatroot.contracts" in chain


def test_package_of_returns_module_for_init():
    """Verify _package_of returns the module itself for __init__.py files.

    An __init__.py IS its package, not a module inside a package.
    """
    init_path = SRC / "trusted" / "__init__.py"
    assert _package_of(init_path) == "beatroot.trusted"


def test_package_of_returns_parent_for_regular_modules():
    """Verify _package_of returns the parent package for regular modules."""
    catalog_path = SRC / "trusted" / "catalog.py"
    assert _package_of(catalog_path) == "beatroot.trusted"


def test_relative_import_from_init_with_correct_package(tmp_path):
    """Verify that `from . import X` inside __init__.py produces correct edge.

    If _package_of is broken, `from . import catalog` in trusted/__init__.py
    (which would be beatroot.trusted package) produces edge "beatroot.catalog"
    instead of "beatroot.trusted.catalog".

    This test uses real files to verify the resolution.
    """
    root = tmp_path / "beatroot"
    root.mkdir()

    trusted = root / "trusted"
    trusted.mkdir()
    (trusted / "__init__.py").write_text("from . import catalog\n")
    (trusted / "catalog.py").write_text("")

    # Build the imports from the __init__.py
    init_file = trusted / "__init__.py"
    imports = _imports(init_file, root)

    # Should contain beatroot.trusted.catalog, not beatroot.catalog
    assert "beatroot.trusted.catalog" in imports, f"Got {imports}"


def test_reachability_survives_a_relative_import_inside_package_init(tmp_path):
    """End-to-end: a relative import *inside* a package __init__.py must not
    break transitive reachability through _internal_graph + _reaches.

    Round 2's test (test_reachability_survives_a_package_init_hop) only used
    `import beatroot.contracts` and `from beatroot import reasoning` — both
    absolute, neither touching `_package_of`'s relative-import arithmetic —
    so it never exercised round 3's bug at all. This builds the specific
    shape that does: `contracts/__init__.py` reaches `reasoning` through a
    RELATIVE `from . import bridge`, and that hop must survive both the
    directory boundary (t0_invariants -> contracts) and the relative-import
    boundary (contracts/__init__.py -> contracts/bridge.py) before landing
    on reasoning.

    Under the round-2 arithmetic (`package = _module_name(path,
    root).rsplit(".", 1)[0]` applied uniformly, including to __init__.py
    files), `from . import bridge` inside contracts/__init__.py resolves
    one level too high, to `beatroot.bridge` instead of
    `beatroot.contracts.bridge`. `beatroot.bridge` is not a real module, and
    `_resolve_edge_to_node` falls back to its parent `beatroot` (the root
    __init__.py), which is a real, import-free graph node. The chain
    dead-ends there instead of reaching bridge.py's `from ...reasoning
    import llm`, and `_reaches` returns None — the T0 -> reasoning path
    silently vanishes.
    """
    root = tmp_path / "beatroot"
    root.mkdir()
    (root / "__init__.py").write_text("")

    t0 = root / "t0_invariants"
    t0.mkdir()
    (t0 / "__init__.py").write_text("")
    (t0 / "constraints.py").write_text("import beatroot.contracts\n")

    contracts = root / "contracts"
    contracts.mkdir()
    (contracts / "__init__.py").write_text("from . import bridge\n")
    (contracts / "bridge.py").write_text("from ...reasoning import llm\n")

    reasoning = root / "reasoning"
    reasoning.mkdir()
    (reasoning / "__init__.py").write_text("")
    (reasoning / "llm.py").write_text("")

    graph = _internal_graph(root)
    chain = _reaches("beatroot.t0_invariants.constraints", "beatroot.reasoning", graph)
    assert chain is not None, (
        "relative import inside contracts/__init__.py broke the transitive chain to reasoning"
    )
    assert "beatroot.contracts" in chain
    assert "beatroot.contracts.bridge" in chain


def test_module_name_canonicalises_package_init():
    """Verify __init__.py files are canonicalized to package names."""
    assert _module_name(SRC / "trusted" / "__init__.py") == "beatroot.trusted"
    assert _module_name(SRC / "trusted" / "catalog.py") == "beatroot.trusted.catalog"


def test_internal_graph_detects_collisions(tmp_path):
    """Verify _internal_graph raises on duplicate module names, and that the
    message names both offending files.

    A `trusted/` package directory and a sibling `trusted.py` module coexist
    fine on the filesystem — nothing prevents it — and both resolve to the
    module name `beatroot.trusted`. That is the real collision this guards
    against, so the test must actually construct it rather than trust a
    comment that claims the filesystem rules it out.
    """
    root = tmp_path / "beatroot"
    root.mkdir()

    trusted_pkg = root / "trusted"
    trusted_pkg.mkdir()
    trusted_init = trusted_pkg / "__init__.py"
    trusted_init.write_text("")

    trusted_module = root / "trusted.py"
    trusted_module.write_text("")

    with pytest.raises(ValueError) as excinfo:
        _internal_graph(root)

    message = str(excinfo.value)
    assert str(trusted_init) in message, f"first file missing from message: {message}"
    assert str(trusted_module) in message, f"second file missing from message: {message}"


def test_nutrition_facts_cannot_be_constructed_from_model_output():
    """Grep-level guarantee: no module builds NutritionFacts with a non-computed
    provenance. The Literal type already forbids it; this catches a future
    author widening the type."""
    from beatroot.contracts.nutrition import NutritionFacts

    field = NutritionFacts.model_fields["provenance"]
    assert field.annotation.__args__ == ("computed",)


def test_packages_under_test_are_non_empty():
    """Guard against the boundary test silently passing because a package moved."""
    for package in FORBIDDEN:
        assert _modules(package), f"no modules found under {package}/"


def test_llm_permitted_false_skills_have_no_call_path_to_reasoning():
    """`llm_permitted: false` is a reachability claim about a specific
    module, not just a YAML field. README.md and ARCHITECTURE.md both
    claim a test verifies every skill declaring `llm_permitted: false` has
    no call path into `reasoning/` — `tests/agent/test_skills_registry.py::
    test_t0_skills_declare_no_llm` only asserted the frontmatter field
    equals `False`, a self-consistency check on a string, not a boundary
    proof. This is the real one: it maps each `llm_permitted: false` skill
    to the module that actually implements it and runs the same
    canonicalised-BFS reachability check the tier-level test above uses,
    against `beatroot.reasoning`.
    """
    from beatroot.agent.skills_registry import DEFAULT_SKILLS_DIR, load_skills

    skills = load_skills(DEFAULT_SKILLS_DIR)

    # The module that actually implements each `llm_permitted: false`
    # skill — not the package it lives under (already covered, at package
    # granularity, by the tests above); this pins the claim to the exact
    # file a reader would open to check it by hand.
    skill_to_module = {
        "check_feasibility": "beatroot.t0_invariants.feasibility",
        "compute_nutrition": "beatroot.t0_invariants.nutrition_math",
    }
    llm_permitted_false = {sid for sid, skill in skills.items() if skill.llm_permitted is False}
    assert llm_permitted_false == set(skill_to_module), (
        "the set of llm_permitted: false skills changed — update "
        "skill_to_module (and this test) to match, don't let the mapping "
        "silently go stale"
    )

    graph = _internal_graph()
    for skill_id, module in skill_to_module.items():
        assert module in graph, f"{module} (implementing skill {skill_id!r}) not found in graph"
        chain = _reaches(module, "beatroot.reasoning", graph)
        assert chain is None, (
            f"skill {skill_id!r} (llm_permitted: false) implementing module "
            f"{module} reaches beatroot.reasoning through: " + " -> ".join(chain or [])
        )
