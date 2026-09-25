"""LOT1 tests: independence bookkeeping + structural cases (requirements 7, 8)."""

import pytest

from research.p3b.generator import generate_corpus
from research.p3b.model import (
    AggregateSpec,
    DuplicateFrameIdError,
    EpochSpec,
    FrameSpec,
    GroupSpec,
    ScenarioSpec,
    UnknownConstituentError,
    compute_bookkeeping,
)
from tests.p3b.conftest import make_sensor


# ---------------------------------------------------------------------------
# Requirement 7 — the two trap configurations
# ---------------------------------------------------------------------------

def test_twenty_frames_one_session_is_not_twenty_groups(tmp_path):
    # 20 light frames, all in ONE group (one session/lineage).
    sensor = make_sensor(shape=(48, 64))
    frames = tuple(
        FrameSpec(frame_id=f"f{i}", frame_type="light", epoch_id="e0", group_id="g0", ordinal=i)
        for i in range(20)
    )
    scenario = ScenarioSpec(
        name="trap-a", seed=3, sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
    )
    bk = compute_bookkeeping(scenario)
    assert bk.frame_count == 20
    assert bk.observation_count == 20
    # The whole point: 20 frames but ONE independent group, NOT 20.
    assert bk.independent_group_count == 1
    assert bk.epoch_count == 1


def test_aggregate_does_not_add_independent_observation(tmp_path):
    # 20 light frames in one group, plus a derived master (aggregate) of them.
    sensor = make_sensor(shape=(48, 64))
    frames = tuple(
        FrameSpec(frame_id=f"f{i}", frame_type="light", epoch_id="e0", group_id="g0", ordinal=i)
        for i in range(20)
    )
    scenario = ScenarioSpec(
        name="trap-b", seed=3, sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
        aggregates=(AggregateSpec(
            aggregate_id="master0", method="median",
            constituent_frame_ids=tuple(f.frame_id for f in frames),
        ),),
    )
    bk = compute_bookkeeping(scenario)
    # The master is a frame (frame_count +1) but NOT a new observation and NOT
    # a new independent group.
    assert bk.frame_count == 21
    assert bk.observation_count == 20
    assert bk.independent_group_count == 1
    assert bk.epoch_count == 1

    # And it actually materialises.
    r = generate_corpus(scenario, tmp_path / "c")
    assert "master0" in r.frame_paths
    assert r.bookkeeping.frame_count == 21
    assert r.bookkeeping.observation_count == 20
    assert r.bookkeeping.independent_group_count == 1


# ---------------------------------------------------------------------------
# Requirement 8 — the six structural cases
# ---------------------------------------------------------------------------

def _mk(frames_per_group, groups_per_epoch, epochs, frame_type="light", seed=1, shape=(48, 64)):
    sensor = make_sensor(shape=shape)
    epoch_specs = []
    counter = 0
    for e in range(epochs):
        groups = []
        for g in range(groups_per_epoch):
            fr = tuple(
                FrameSpec(
                    frame_id=f"f{counter + i}",
                    frame_type=frame_type,
                    epoch_id=f"e{e}",
                    group_id=f"g{e}_{g}",
                    ordinal=i,
                )
                for i in range(frames_per_group)
            )
            counter += frames_per_group
            groups.append(GroupSpec(group_id=f"g{e}_{g}", frames=fr))
        epoch_specs.append(EpochSpec(epoch_id=f"e{e}", groups=tuple(groups)))
    return ScenarioSpec(name="structural", seed=seed, sensor=sensor, epochs=tuple(epoch_specs))


def test_case_1_epoch_many_frames():
    bk = compute_bookkeeping(_mk(frames_per_group=8, groups_per_epoch=1, epochs=1))
    assert bk.epoch_count == 1
    assert bk.frame_count == 8
    assert bk.observation_count == 8
    assert bk.independent_group_count == 1


def test_case_2_epochs_one_group_each():
    bk = compute_bookkeeping(_mk(frames_per_group=4, groups_per_epoch=1, epochs=2))
    assert bk.epoch_count == 2
    assert bk.frame_count == 8
    assert bk.independent_group_count == 2


def test_case_2_epochs_multiple_groups():
    bk = compute_bookkeeping(_mk(frames_per_group=3, groups_per_epoch=3, epochs=2))
    assert bk.epoch_count == 2
    assert bk.frame_count == 18
    assert bk.independent_group_count == 6


def test_case_many_frames_same_lineage():
    bk = compute_bookkeeping(_mk(frames_per_group=12, groups_per_epoch=1, epochs=1))
    assert bk.frame_count == 12
    assert bk.independent_group_count == 1  # many frames, one lineage


def test_case_master_plus_own_constituents():
    sensor = make_sensor(shape=(48, 64))
    darks = tuple(
        FrameSpec(frame_id=f"d{i}", frame_type="dark", epoch_id="e0", group_id="gd", ordinal=i)
        for i in range(5)
    )
    scenario = ScenarioSpec(
        name="master-constituents", seed=2, sensor=sensor,
        epochs=(EpochSpec("e0", (GroupSpec("gd", darks),)),),
        aggregates=(AggregateSpec(
            aggregate_id="darkmaster", method="median",
            constituent_frame_ids=tuple(f.frame_id for f in darks),
        ),),
    )
    bk = compute_bookkeeping(scenario)
    # 5 dark frames + 1 master = 6 frames; darks are calibration (not light
    # observations), so observation_count=0 and there is no science group.
    assert bk.frame_count == 6
    assert bk.observation_count == 0
    assert bk.independent_group_count == 0
    assert bk.epoch_count == 1


def test_case_master_plus_independent_replacement_sequence():
    sensor = make_sensor(shape=(48, 64))
    # Lineage A: a light sequence that gets a master.
    lights_a = tuple(
        FrameSpec(frame_id=f"a{i}", frame_type="light", epoch_id="e0", group_id="ga", ordinal=i)
        for i in range(4)
    )
    # Independent replacement sequence B (separate group, separate epoch).
    lights_b = tuple(
        FrameSpec(frame_id=f"b{i}", frame_type="light", epoch_id="e1", group_id="gb", ordinal=i)
        for i in range(4)
    )
    scenario = ScenarioSpec(
        name="master-replacement", seed=4, sensor=sensor,
        epochs=(
            EpochSpec("e0", (GroupSpec("ga", lights_a),)),
            EpochSpec("e1", (GroupSpec("gb", lights_b),)),
        ),
        aggregates=(AggregateSpec(
            aggregate_id="master_a", method="median",
            constituent_frame_ids=tuple(f.frame_id for f in lights_a),
        ),),
    )
    bk = compute_bookkeeping(scenario)
    # The master of A adds a frame but not an observation/group; B is an
    # independent replacement lineage (its own group + epoch).
    assert bk.frame_count == 9  # 8 lights + 1 master
    assert bk.observation_count == 8
    assert bk.independent_group_count == 2
    assert bk.epoch_count == 2


# ---------------------------------------------------------------------------
# Rework-1 (M1): independent-group identity is the (epoch_id, group_id) PAIR
# ---------------------------------------------------------------------------

def test_same_group_id_in_two_epochs_counts_two_groups():
    # Two epochs, one group each, BOTH named "g0". They must not collapse into
    # one independent group.
    sensor = make_sensor(shape=(48, 64))
    scenario = ScenarioSpec(
        name="collide", seed=1, sensor=sensor,
        epochs=(
            EpochSpec("e0", (GroupSpec("g0", (
                FrameSpec("a", "light", "e0", "g0", 0),
            ),),)),
            EpochSpec("e1", (GroupSpec("g0", (
                FrameSpec("b", "light", "e1", "g0", 0),
            ),),)),
        ),
    )
    bk = compute_bookkeeping(scenario)
    assert bk.frame_count == 2
    assert bk.observation_count == 2
    assert bk.independent_group_count == 2
    assert bk.epoch_count == 2


def test_same_group_ids_reused_across_epochs_count_correctly():
    # 2 epochs x 2 groups, ids reused across epochs -> 4 independent groups.
    sensor = make_sensor(shape=(48, 64))
    epochs = []
    counter = 0
    for e in (0, 1):
        groups = []
        for g in (0, 1):
            groups.append(GroupSpec(f"g{g}", (
                FrameSpec(f"f{counter}", "light", f"e{e}", f"g{g}", 0),
            ),))
            counter += 1
        epochs.append(EpochSpec(f"e{e}", tuple(groups)))
    scenario = ScenarioSpec(name="multi-collide", seed=2, sensor=sensor, epochs=tuple(epochs))
    bk = compute_bookkeeping(scenario)
    assert bk.frame_count == 4
    assert bk.observation_count == 4
    assert bk.independent_group_count == 4
    assert bk.epoch_count == 2


def test_manifest_serializes_corrected_bookkeeping(tmp_path):
    # The manifest (not just the function) must carry the corrected counters.
    sensor = make_sensor(shape=(48, 64))
    scenario = ScenarioSpec(
        name="collide", seed=1, sensor=sensor,
        epochs=(
            EpochSpec("e0", (GroupSpec("g0", (
                FrameSpec("a", "light", "e0", "g0", 0),
            ),),)),
            EpochSpec("e1", (GroupSpec("g0", (
                FrameSpec("b", "light", "e1", "g0", 0),
            ),),)),
        ),
    )
    r = generate_corpus(scenario, tmp_path / "c")
    assert r.manifest["bookkeeping"]["independent_group_count"] == 2
    assert r.manifest["bookkeeping"]["epoch_count"] == 2
    assert r.manifest["bookkeeping"]["frame_count"] == 2
    assert r.manifest["bookkeeping"]["observation_count"] == 2


# ---------------------------------------------------------------------------
# Rework-1 (M2): typed identifier validation
# ---------------------------------------------------------------------------

def test_duplicate_frame_id_raises_typed_error():
    sensor = make_sensor(shape=(48, 64))
    with pytest.raises(DuplicateFrameIdError):
        ScenarioSpec(
            name="dup", seed=1, sensor=sensor,
            epochs=(
                EpochSpec("e0", (GroupSpec("g0", (
                    FrameSpec("same", "light", "e0", "g0", 0),
                ),),)),
                EpochSpec("e1", (GroupSpec("g1", (
                    FrameSpec("same", "light", "e1", "g1", 0),
                ),),)),
            ),
        )


def test_duplicate_frame_id_across_frame_types_raises():
    sensor = make_sensor(shape=(48, 64))
    with pytest.raises(DuplicateFrameIdError):
        ScenarioSpec(
            name="dup-type", seed=1, sensor=sensor,
            epochs=(
                EpochSpec("e0", (
                    GroupSpec("g0", (FrameSpec("same", "light", "e0", "g0", 0),)),
                    GroupSpec("gd", (FrameSpec("same", "dark", "e0", "gd", 0),)),
                )),
            ),
        )


def test_aggregate_with_unknown_constituent_raises_typed_error():
    sensor = make_sensor(shape=(48, 64))
    with pytest.raises(UnknownConstituentError):
        ScenarioSpec(
            name="orphan", seed=1, sensor=sensor,
            epochs=(EpochSpec("e0", (GroupSpec("g0", (
                FrameSpec("a", "light", "e0", "g0", 0),
            ),),)),),
            aggregates=(AggregateSpec(
                aggregate_id="m", method="median", constituent_frame_ids=("a", "ghost"),
            ),),
        )


def test_duplicate_aggregate_id_raises_typed_error():
    sensor = make_sensor(shape=(48, 64))
    frames = (FrameSpec("a", "light", "e0", "g0", 0),)
    with pytest.raises(DuplicateFrameIdError):
        ScenarioSpec(
            name="dup-agg", seed=1, sensor=sensor,
            epochs=(EpochSpec("e0", (GroupSpec("g0", frames),)),),
            aggregates=(
                AggregateSpec("m", "median", ("a",)),
                AggregateSpec("m", "median", ("a",)),
            ),
        )
