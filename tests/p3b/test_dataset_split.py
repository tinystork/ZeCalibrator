"""LOT7 tests: dataset separation + anti-leakage guard (properties 1, 2, 4, 5, 6, 7).

Property 3 (holdout independence) is tested separately in
``tests/p3b/test_holdout.py``.
"""

import ast
import inspect

import pytest

import research.p3b.dataset_split as dataset_split
from research.p3b.dataset_split import (
    DATASET_DEVELOPMENT,
    DATASET_HOLDOUT,
    DATASET_QUALIFICATION,
    DATASET_ROLES,
    DatasetRegistry,
    DuplicateScenarioError,
    SignatureCollisionError,
    build_manifest,
    scenario_signature,
)
from research.p3b.model import (
    AggregateSpec,
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    SiteSpec,
)
from tests.p3b.conftest import make_sensor


def _site_scenario(name, seed, sensor=None):
    """A small deterministic scenario: 3 light frames + 2 declared sites.

    Two calls with the same ``seed``/``sensor``/sites but different ``name``
    produce the *same signature* (the name is a label, not part of what the
    data is). Different seeds produce distinct signatures.
    """
    sensor = sensor or make_sensor(shape=(48, 64))
    sites = (
        SiteSpec(site_id="s0", x=10, y=12, cfa_class="STABLE_ANOMALY_WITH_MISMATCHED_DARK"),
        SiteSpec(site_id="s1", x=20, y=22, cfa_class="NORMAL"),
    )
    frames = tuple(
        FrameSpec(frame_id=f"l{i}", frame_type="light", epoch_id="e0", group_id="g0", ordinal=i)
        for i in range(3)
    )
    return ScenarioSpec(
        name=name,
        seed=seed,
        sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
        sites=sites,
    )


# ---------------------------------------------------------------------------
# Property 1 — effective separation + reproducible manifests
# ---------------------------------------------------------------------------

def test_development_and_holdout_distinct_seeds_give_distinct_signatures():
    dev = _site_scenario("dev", seed=11)
    hold = _site_scenario("hold", seed=22)
    assert scenario_signature(dev) != scenario_signature(hold)


def test_manifest_reproducible_same_hash_on_regeneration():
    scenarios = (_site_scenario("dev", 3), _site_scenario("qual", 4))
    m1 = build_manifest(DATASET_DEVELOPMENT, scenarios)
    m2 = build_manifest(DATASET_DEVELOPMENT, scenarios)
    assert m1.manifest_hash == m2.manifest_hash
    assert m1.to_dict() == m2.to_dict()
    assert [r.signature for r in m1.scenarios] == [r.signature for r in m2.scenarios]


def test_three_roles_are_distinct_and_defined():
    assert DATASET_ROLES == (
        DATASET_DEVELOPMENT,
        DATASET_QUALIFICATION,
        DATASET_HOLDOUT,
    )
    assert len(set(DATASET_ROLES)) == 3


# ---------------------------------------------------------------------------
# Property 2 — leakage refused (typed), not a silent warning
# ---------------------------------------------------------------------------

def test_same_signature_across_two_datasets_raises_typed_error():
    registry = DatasetRegistry()
    dev = _site_scenario("dev", seed=7)
    hold = _site_scenario("hold", seed=7)
    # Same declared science, different label -> same signature.
    assert scenario_signature(dev) == scenario_signature(hold)

    registry.register(DATASET_DEVELOPMENT, (dev,))
    with pytest.raises(SignatureCollisionError) as excinfo:
        registry.register(DATASET_HOLDOUT, (hold,))

    # It is a typed refusal that names the two conflicting datasets.
    assert excinfo.value.first_role == DATASET_DEVELOPMENT
    assert excinfo.value.second_role == DATASET_HOLDOUT


def test_signature_collision_error_is_a_value_error_subtype():
    assert issubclass(SignatureCollisionError, ValueError)
    # Not a warning: it is an exception, never a silent pass-through.
    assert not issubclass(SignatureCollisionError, Warning)


def test_failed_cross_dataset_register_builds_no_manifest():
    registry = DatasetRegistry()
    registry.register(DATASET_DEVELOPMENT, (_site_scenario("dev", 7),))
    with pytest.raises(SignatureCollisionError):
        registry.register(DATASET_HOLDOUT, (_site_scenario("hold", 7),))
    with pytest.raises(KeyError):
        registry.manifest(DATASET_HOLDOUT)


def test_duplicate_within_same_dataset_raises_duplicate_error():
    registry = DatasetRegistry()
    registry.register(DATASET_DEVELOPMENT, (_site_scenario("dev", 7),))
    with pytest.raises(DuplicateScenarioError):
        registry.register(DATASET_DEVELOPMENT, (_site_scenario("dev", 7),))


def test_distinct_signatures_across_datasets_are_allowed():
    registry = DatasetRegistry()
    registry.register(DATASET_DEVELOPMENT, (_site_scenario("dev", 1),))
    registry.register(DATASET_QUALIFICATION, (_site_scenario("qual", 2),))
    registry.register(DATASET_HOLDOUT, (_site_scenario("hold", 3),))
    assert len(registry.manifests()) == 3


# ---------------------------------------------------------------------------
# Property 4 — counter distinction (frame / observation / group / epoch)
# ---------------------------------------------------------------------------

def test_aggregate_adds_frame_not_observation_or_group():
    sensor = make_sensor(shape=(48, 64))
    frames = tuple(
        FrameSpec(frame_id=f"f{i}", frame_type="light", epoch_id="e0", group_id="g0", ordinal=i)
        for i in range(20)
    )
    scenario = ScenarioSpec(
        name="agg",
        seed=1,
        sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
        aggregates=(AggregateSpec("master0", "median", tuple(f.frame_id for f in frames)),),
    )
    record = build_manifest(DATASET_DEVELOPMENT, (scenario,)).scenarios[0]
    # The master is a frame (+1) but NOT an observation and NOT an independent group.
    assert record.frame_count == 21
    assert record.observation_count == 20
    assert record.independent_group_count == 1
    assert record.epoch_count == 1


def test_adding_frames_does_not_increase_independence():
    def scenario_with(n):
        frames = tuple(
            FrameSpec(frame_id=f"f{i}", frame_type="light", epoch_id="e0", group_id="g0", ordinal=i)
            for i in range(n)
        )
        return ScenarioSpec(
            name="f",
            seed=1,
            sensor=make_sensor(shape=(48, 64)),
            epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
        )

    small = build_manifest(DATASET_DEVELOPMENT, (scenario_with(3),)).scenarios[0]
    big = build_manifest(DATASET_DEVELOPMENT, (scenario_with(12),)).scenarios[0]
    assert big.frame_count > small.frame_count
    assert big.observation_count > small.observation_count
    # Adding frames to the same sequence never increases independence.
    assert big.independent_group_count == small.independent_group_count == 1
    assert big.epoch_count == small.epoch_count == 1


def test_counter_fields_are_distinct_and_not_conflated():
    # A scenario where the counters genuinely differ: 2 epochs x 1 group,
    # 3 frames each = 6 frames/observations, 2 groups, 2 epochs, + 1 master.
    sensor = make_sensor(shape=(48, 64))
    epochs = []
    counter = 0
    for e in (0, 1):
        frames = tuple(
            FrameSpec(frame_id=f"f{counter + i}", frame_type="light",
                      epoch_id=f"e{e}", group_id=f"g{e}", ordinal=i)
            for i in range(3)
        )
        counter += 3
        epochs.append(EpochSpec(f"e{e}", (GroupSpec(f"g{e}", frames),)))
    scenario = ScenarioSpec(
        name="multi",
        seed=2,
        sensor=sensor,
        epochs=tuple(epochs),
        aggregates=(AggregateSpec("master0", "median", ("f0", "f1", "f2")),),
    )
    record = build_manifest(DATASET_DEVELOPMENT, (scenario,)).scenarios[0]
    assert record.frame_count == 7
    assert record.observation_count == 6
    assert record.independent_group_count == 2
    assert record.epoch_count == 2
    # The four counters are distinct fields with distinct meanings.
    fields = {
        "frame_count": record.frame_count,
        "observation_count": record.observation_count,
        "independent_group_count": record.independent_group_count,
        "epoch_count": record.epoch_count,
    }
    assert set(fields) == {"frame_count", "observation_count",
                           "independent_group_count", "epoch_count"}


# ---------------------------------------------------------------------------
# Property 5 — no abusive claim of independent real validation
# ---------------------------------------------------------------------------

def test_manifest_never_claims_independent_real_validation():
    manifest = build_manifest(DATASET_HOLDOUT, (_site_scenario("hold", 9),))
    assert manifest.independent_real_validation == "NO"
    assert manifest.real_validation_status == "NOT_SATISFIED"

    d = manifest.to_dict()
    assert d["independent_real_validation"] == "NO"
    assert d["real_validation_status"] == "NOT_SATISFIED"
    # It must never read YES / SATISFIED (the positive claim).
    assert d["independent_real_validation"] != "YES"
    assert d["real_validation_status"] != "SATISFIED"


# ---------------------------------------------------------------------------
# Property 6 — determinism
# ---------------------------------------------------------------------------

def test_same_declaration_same_signature_across_instances():
    a = _site_scenario("x", 5)
    b = _site_scenario("y", 5)
    assert scenario_signature(a) == scenario_signature(b)


def test_repeated_signature_calls_are_stable():
    s = _site_scenario("x", 5)
    assert scenario_signature(s) == scenario_signature(s) == scenario_signature(s)


def test_same_seeds_same_manifest_hash():
    scenarios = (_site_scenario("dev", 8),)
    assert (
        build_manifest(DATASET_DEVELOPMENT, scenarios).manifest_hash
        == build_manifest(DATASET_DEVELOPMENT, scenarios).manifest_hash
    )


# ---------------------------------------------------------------------------
# Property 7 — no threshold, no selection (AST), no acceptance rule
# ---------------------------------------------------------------------------

def _forbidden_identifiers(*modules):
    forbidden = {
        "argmax", "argmin", "sorted", "best", "optimal",
        "threshold", "accept", "acceptance",
    }
    offenders = []
    for mod in modules:
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                offenders.append((mod.__name__, node.id))
            elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                offenders.append((mod.__name__, node.attr))
    return offenders


def test_no_threshold_or_selection_tokens_in_guard_source():
    import research.p3b.holdout as holdout_module

    assert _forbidden_identifiers(dataset_split, holdout_module) == []
