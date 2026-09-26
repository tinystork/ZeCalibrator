"""P4.2.1 BPM UX-hardening GUI tests: detector-K setting + §3/§4 decision flows.

Covers: the Advanced-tab "Bad pixel detection threshold" setting (default 30.0,
persistence/restoration, Reset default, calm validation), the §3 rebuild-vs-use
decision (once per run; Rebuild/Use existing/Cancel), the §4 no-database
Create/Select/Not now flows, folder-picker cancellation safety, and the §7
removal of any user-visible "preview disabled" wording.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from zecalibrator.gui.window import MainWindow
from zecalibrator.storage import resolve_paths


@pytest.fixture
def paths(tmp_path):
    return resolve_paths(base=str(tmp_path))


def _pump(cond, timeout_ms=15000):
    from PySide6 import QtTest

    start = time.monotonic()
    while time.monotonic() - start < timeout_ms / 1000.0:
        QtWidgets.QApplication.processEvents()
        if cond():
            return True
        QtTest.QTest.qWait(5)
    return False


def _close(w):
    w.close()
    _pump(lambda: w._controller.is_finished)


def _fake_message_box(monkeypatch, clicked, capture=None):
    """Monkeypatch QMessageBox with a fake that returns ``clicked`` when asked.

    ``addButton`` returns a stable token object per label; ``clickedButton``
    returns the token for ``clicked`` so the window's identity checks
    (``clicked is use_btn``) behave like the real widget.
    """

    class _Icon:
        Question = "Question"
        Warning = "Warning"

    class _ButtonRole:
        AcceptRole = "AcceptRole"
        RejectRole = "RejectRole"
        DestructiveRole = "DestructiveRole"

    class _Box:
        Icon = _Icon
        ButtonRole = _ButtonRole

        def __init__(self, parent):
            self.parent = parent
            self._tokens = {}

        def setWindowTitle(self, t):
            self.title = t
            if capture is not None:
                capture["title"] = t

        def setIcon(self, i):
            self.icon = i

        def setText(self, t):
            self.text = t
            if capture is not None:
                capture["text"] = t

        def addButton(self, label, role):
            if capture is not None:
                capture.setdefault("buttons", []).append(label)
            self._tokens.setdefault(label, object())
            return self._tokens[label]

        def exec(self):
            pass

        def clickedButton(self):
            if clicked is None:
                return None
            self._tokens.setdefault(clicked, object())
            return self._tokens[clicked]

    monkeypatch.setattr(QtWidgets, "QMessageBox", _Box)
    return _Box


class _FakePlan:
    def __init__(self, *, has_dark=True):
        self.light_constraints = "dummy-light"
        self.masters = {"dark": object()} if has_dark else {}


def _prep(qapp, monkeypatch, tmp_path, decision):
    """Build a window with BPM loaded, a fake dark plan, and a stubbed decision."""
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
    w._bpm_root = str(tmp_path / "base")
    w._plans = {"r1": _FakePlan(has_dark=True)}
    monkeypatch.setattr(_bpm, "bpm_creation_decision", decision)
    return w


# ---------------------------------------------------------------------------
# §1 detector-K setting widget
# ---------------------------------------------------------------------------
def test_bpm_threshold_widget_default_is_30(qapp, paths):
    w = MainWindow(paths)
    try:
        assert w.bpm_k_edit is not None
        assert w.bpm_k_reset_btn is not None
        assert w.bpm_k_edit.value() == 30.0
        assert w.bpm_k_edit.toolTip() == "Lower values detect more candidate bad pixels."
    finally:
        _close(w)


def test_bpm_threshold_reset_default_restores_30(qapp, paths):
    w = MainWindow(paths)
    try:
        assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
        w.bpm_k_edit.setValue(45.0)
        w._bpm_detector_k = 45.0
        w._on_bpm_k_reset()
        assert w._bpm_detector_k == 30.0
        assert w.bpm_k_edit.value() == 30.0
    finally:
        _close(w)


def test_bpm_threshold_loads_from_settings(qapp, tmp_path):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    _bpm.save_bpm_settings(
        paths.user_config_path,
        _bpm.bpm_settings(str(tmp_path / "base"), detector_k=20.0),
    )
    w = MainWindow(paths)
    try:
        assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
        assert w._bpm_detector_k == 20.0
        assert w.bpm_k_edit.value() == 20.0
    finally:
        _close(w)


def test_bpm_threshold_change_persists(qapp, tmp_path):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    w = MainWindow(paths)
    try:
        assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
        w.bpm_k_edit.setValue(25.0)
        assert _pump(lambda: not w._controller.is_active)
        loaded = _bpm.load_bpm_settings(paths.user_config_path)
        assert loaded.settings.detector_k == 25.0
    finally:
        _close(w)


def test_bpm_threshold_invalid_value_is_rejected_calmly(qapp, paths):
    from zecalibrator.api.v1 import _bpm

    w = MainWindow(paths)
    try:
        assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
        before = w._bpm_detector_k
        # A QDoubleSpinBox cannot produce a non-finite value; drive the validation
        # headlessly to prove invalid input is rejected without a crash.
        import pytest as _pytest

        with _pytest.raises(ValueError):
            _bpm.validate_detector_k(0.0)
        with _pytest.raises(ValueError):
            _bpm.validate_detector_k(float("inf"))
        with _pytest.raises(ValueError):
            _bpm.validate_detector_k(-5.0)
        assert w._bpm_detector_k == before  # unchanged after rejected input
    finally:
        _close(w)


# ---------------------------------------------------------------------------
# §3 mismatch decision (once per run)
# ---------------------------------------------------------------------------
_MISMATCH = lambda *a, dark_available=False, **k: {  # noqa: E731
    "action": "mismatch", "reason_code": "", "revision_id": "rev-1",
    "map_k": 30.0, "setting_k": 20.0,
}


def test_mismatch_offer_is_emitted_once_per_run(qapp, monkeypatch, tmp_path):
    w = _prep(qapp, monkeypatch, tmp_path, _MISMATCH)
    try:
        offers = []
        exports = []

        def _offer(request, destination, decision):
            offers.append(decision)
            # The dialog records an explicit decision (e.g. Cancel).
            w._bpm_mismatch_decision = "cancel"

        w._offer_bpm_mismatch = _offer
        w._start_export = lambda request, destination: exports.append((request, destination))

        w._check_bpm_proposal_then_export(None, "/dest")
        assert len(offers) == 1
        assert w._bpm_mismatch_decision == "cancel"
        # A second export in the SAME run must not re-offer (the decision is cached).
        w._check_bpm_proposal_then_export(None, "/dest")
        assert len(offers) == 1
        assert exports == []  # Cancel never silently exports on re-click
    finally:
        _close(w)


def test_mismatch_use_existing_exports_and_never_relabels(qapp, monkeypatch, tmp_path):
    w = _prep(qapp, monkeypatch, tmp_path, _MISMATCH)
    try:
        exports = []
        w._start_export = lambda request, destination: exports.append((request, destination))
        _fake_message_box(monkeypatch, "Use existing map")
        w._offer_bpm_mismatch(None, "/dest", _MISMATCH())
        assert len(exports) == 1
        assert exports[0] == (None, "/dest")
    finally:
        _close(w)


def test_mismatch_cancel_does_not_export(qapp, monkeypatch, tmp_path):
    w = _prep(qapp, monkeypatch, tmp_path, _MISMATCH)
    try:
        exports = []
        w._start_export = lambda request, destination: exports.append((request, destination))
        w._request_bpm_map_creation = lambda request, destination: exports.append(("REBUILD", request))
        _fake_message_box(monkeypatch, "Cancel")
        w._offer_bpm_mismatch(None, "/dest", _MISMATCH())
        # Cancel: no run, no output, no revision.
        assert exports == []
    finally:
        _close(w)


def test_mismatch_rebuild_requests_map_creation(qapp, monkeypatch, tmp_path):
    w = _prep(qapp, monkeypatch, tmp_path, _MISMATCH)
    try:
        requests = []
        w._request_bpm_map_creation = lambda request, destination: requests.append((request, destination))
        w._start_export = lambda request, destination: requests.append(("export", request, destination))
        _fake_message_box(monkeypatch, "Rebuild")
        w._offer_bpm_mismatch(None, "/dest", _MISMATCH())
        assert requests == [(None, "/dest")]
    finally:
        _close(w)


def test_mismatch_without_dark_never_offers_rebuild(qapp, monkeypatch, tmp_path):
    # No selected compatible Master Dark -> Rebuild is not offered (calm, honest).
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
    w._bpm_root = str(tmp_path / "base")
    w._plans = {"r1": _FakePlan(has_dark=False)}
    monkeypatch.setattr(_bpm, "bpm_creation_decision", _MISMATCH)

    seen = {}
    _fake_message_box(monkeypatch, "Cancel", capture=seen)
    try:
        w._offer_bpm_mismatch(None, "/dest", _MISMATCH())
        assert "Rebuild" not in seen.get("buttons", [])
        assert "Use existing map" in seen.get("buttons", [])
        assert "Cancel" in seen.get("buttons", [])
    finally:
        _close(w)


# ---------------------------------------------------------------------------
# §3 F2 regression: the mismatch decision is an explicit per-run CACHE.
# ---------------------------------------------------------------------------
def test_mismatch_cancel_is_sticky_for_the_run(qapp, monkeypatch, tmp_path):
    """F2: Cancel caches an explicit decision; a second export in the same
    unchanged run must NOT silently export with the old map."""
    w = _prep(qapp, monkeypatch, tmp_path, _MISMATCH)
    try:
        exports = []
        w._start_export = lambda request, destination: exports.append((request, destination))
        # First click: Cancel.
        _fake_message_box(monkeypatch, "Cancel")
        w._check_bpm_proposal_then_export(None, "/dest")
        assert w._bpm_mismatch_decision == "cancel"
        assert exports == []
        # Second click in the SAME unchanged run: still cancelled, no dialog, no export.
        dialogs = []
        w._offer_bpm_mismatch = lambda request, destination, decision: dialogs.append(decision)
        w._check_bpm_proposal_then_export(None, "/dest")
        assert dialogs == []  # no second dialog
        assert exports == []  # no silent "Use existing"
    finally:
        _close(w)


def test_mismatch_use_existing_sticky_no_reprompt(qapp, monkeypatch, tmp_path):
    """F2: Use existing is cached; later exports reuse it without re-prompting."""
    w = _prep(qapp, monkeypatch, tmp_path, _MISMATCH)
    try:
        exports = []
        dialogs = []
        w._start_export = lambda request, destination: exports.append((request, destination))
        _fake_message_box(monkeypatch, "Use existing map")
        w._check_bpm_proposal_then_export(None, "/dest")
        assert w._bpm_mismatch_decision == "use_existing"
        assert len(exports) == 1
        # Second export: reuses the decision, no second dialog.
        real_offer = w._offer_bpm_mismatch
        w._offer_bpm_mismatch = lambda request, destination, decision: dialogs.append(decision) or real_offer(
            request, destination, decision
        )
        w._check_bpm_proposal_then_export(None, "/dest")
        assert dialogs == []  # no re-prompt
        assert len(exports) == 2
    finally:
        _close(w)


def test_mismatch_generation_bump_resets_decision(qapp, monkeypatch, tmp_path):
    """F2: a fresh input/config generation resets the cached decision and allows
    one new prompt."""
    w = _prep(qapp, monkeypatch, tmp_path, _MISMATCH)
    try:
        exports = []
        w._start_export = lambda request, destination: exports.append((request, destination))
        _fake_message_box(monkeypatch, "Cancel")
        w._check_bpm_proposal_then_export(None, "/dest")
        assert w._bpm_mismatch_decision == "cancel"
        assert exports == []

        # A config/input change begins a new run.
        w._bump_generation()
        w._plans = {"r1": _FakePlan(has_dark=True)}
        assert w._bpm_mismatch_decision is None
        # One new prompt is allowed again.
        dialogs = []
        w._offer_bpm_mismatch = lambda request, destination, decision: dialogs.append(decision)
        w._check_bpm_proposal_then_export(None, "/dest")
        assert len(dialogs) == 1
        assert exports == []
    finally:
        _close(w)


def test_mismatch_dialog_closed_equals_cancel(qapp, monkeypatch, tmp_path):
    """F2: closing the dialog (clickedButton() is None) is equivalent to Cancel."""
    w = _prep(qapp, monkeypatch, tmp_path, _MISMATCH)
    try:
        exports = []
        w._start_export = lambda request, destination: exports.append((request, destination))
        _fake_message_box(monkeypatch, None)  # dialog closed
        w._check_bpm_proposal_then_export(None, "/dest")
        assert w._bpm_mismatch_decision == "cancel"
        assert exports == []
        # And it stays cancelled for the run.
        w._check_bpm_proposal_then_export(None, "/dest")
        assert exports == []
    finally:
        _close(w)


# ---------------------------------------------------------------------------
# §4 no-database decision (Create / Select / Not now)
# ---------------------------------------------------------------------------
def test_no_db_offer_create_select_not_now_once(qapp, monkeypatch, tmp_path):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
    w._bpm_root = None  # no DB configured
    w._plans = {"r1": _FakePlan(has_dark=True)}
    monkeypatch.setattr(_bpm, "bpm_no_database_decision", lambda s: {"configured": False, "root": None})

    offers = []
    exports = []
    w._offer_bpm_database_setup = lambda request, destination: offers.append((request, destination))
    w._start_export = lambda request, destination: exports.append((request, destination))

    w._check_bpm_proposal_then_export(None, "/dest")
    assert len(offers) == 1
    assert w._bpm_no_db_prompted is True
    w._check_bpm_proposal_then_export(None, "/dest")
    assert len(offers) == 1  # once per run


def test_no_db_create_persists_root_then_proposes_map(qapp, monkeypatch, tmp_path):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    new_base = tmp_path / "new" / "base"
    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
    w._bpm_root = None
    w._plans = {"r1": _FakePlan(has_dark=True)}
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: str(new_base)),
    )
    # After Create, the SAME workflow runs: no compatible map + dark -> propose.
    monkeypatch.setattr(_bpm, "bpm_creation_decision",
                        lambda *a, dark_available=False, **k: {"action": "propose", "reason_code": "NO_PROFILE"})

    proposal = []
    w._offer_bpm_creation = lambda request, destination: proposal.append((request, destination))
    w._start_export = lambda request, destination: None

    try:
        w._choose_bpm_database(None, "/dest", create=True)
        assert new_base.is_dir()
        assert w._bpm_root == str(new_base)
        assert _pump(lambda: not w._controller.is_active)
        loaded = _bpm.load_bpm_settings(paths.user_config_path)
        assert loaded.settings.bad_pixel_database_root == str(new_base)
        # The map proposal was reached after root persistence (once).
        assert len(proposal) == 1
    finally:
        _close(w)


def test_no_db_select_existing_lookup_automatic(qapp, monkeypatch, tmp_path):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    existing = tmp_path / "existing"
    existing.mkdir()
    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
    w._bpm_root = None
    w._plans = {"r1": _FakePlan(has_dark=True)}
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: str(existing)),
    )
    # After Select, lookup finds a compatible map -> use automatically.
    monkeypatch.setattr(_bpm, "bpm_creation_decision",
                        lambda *a, dark_available=False, **k: {"action": "use", "reason_code": "", "map_k": 30.0})
    exports = []
    w._start_export = lambda request, destination: exports.append((request, destination))
    w._offer_bpm_creation = lambda request, destination: None

    try:
        w._choose_bpm_database(None, "/dest", create=False)
        assert w._bpm_root == str(existing)
        assert _pump(lambda: not w._controller.is_active)
        loaded = _bpm.load_bpm_settings(paths.user_config_path)
        assert loaded.settings.bad_pixel_database_root == str(existing)
        assert len(exports) == 1  # lookup automatic -> export proceeds
    finally:
        _close(w)


def test_no_db_not_now_normal_calibration_no_error(qapp, monkeypatch, tmp_path):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
    w._bpm_root = None
    w._plans = {"r1": _FakePlan(has_dark=True)}
    monkeypatch.setattr(_bpm, "bpm_no_database_decision", lambda s: {"configured": False, "root": None})

    exports = []
    _fake_message_box(monkeypatch, "Not now")
    w._start_export = lambda request, destination: exports.append((request, destination))
    try:
        w._offer_bpm_database_setup(None, "/dest")
        assert len(exports) == 1  # normal calibration, no error
        assert w._bpm_root is None  # still unconfigured
    finally:
        _close(w)


def test_folder_picker_cancel_is_calm_and_safe(qapp, monkeypatch, tmp_path):
    from zecalibrator.api.v1 import _bpm

    paths = resolve_paths(base=str(tmp_path))
    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
    w._bpm_root = None
    w._plans = {"r1": _FakePlan(has_dark=True)}
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))  # user cancelled

    exports = []
    w._start_export = lambda request, destination: exports.append((request, destination))
    try:
        w._choose_bpm_database(None, "/dest", create=True)
        # Cancellation falls back to normal calibration; no partial BPM path.
        assert len(exports) == 1
        assert w._bpm_root is None
    finally:
        _close(w)


# ---------------------------------------------------------------------------
# §7 no user-visible "preview disabled" wording
# ---------------------------------------------------------------------------
def test_no_preview_disabled_wording_in_status(qapp, paths):
    w = MainWindow(paths)
    try:
        assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)
        assert w.bpm_root_edit.placeholderText() == "No Bad Pixel Database configured"
        assert "preview disabled" not in w.bpm_status_label.text()
        assert "preview" not in w.bpm_status_label.text().lower()
    finally:
        _close(w)


# ---------------------------------------------------------------------------
# F1 regression: fresh-DB onboarding serializes save → offer/create → export
# using the REAL controller/worker (no stubbed continuation).
# ---------------------------------------------------------------------------
def test_fresh_db_onboarding_serializes_save_then_create_then_export(qapp, monkeypatch, tmp_path):
    """F1: after Create root, map creation must be admitted only AFTER the
    save_bpm_settings worker ends; the operation ordering is save → create map
    → export, with no "operation is already running" and a real revision."""
    import json
    from zecalibrator.api.v1 import _bpm
    import zecalibrator.api.v1 as v1
    from zecalibrator.gui.window import _LightEntry
    from .conftest import make_synth_fixture

    paths = resolve_paths(base=str(tmp_path))
    fixture = make_synth_fixture(tmp_path)
    new_base = tmp_path / "new" / "base"

    w = MainWindow(paths)
    assert _pump(lambda: w._bpm_loaded and not w._controller.is_active)

    # Real light + library (dark master indexed by make_synth_fixture).
    decl = v1.ImportDeclaration(**json.loads(open(fixture["decl"]).read()))
    roi = v1.RoiExtentEvidence(**json.loads(open(fixture["roi"]).read()))
    w._lights.append(_LightEntry(fixture["light"], hdu=0, declaration=decl, roi_extent=roi))
    w._refresh_lights_list()
    w._library_spec = v1.LibrarySpec(root=fixture["root"], index_path=fixture["index"])

    # Real preflight to populate a resolved plan (dark binding).
    w._on_preflight()
    assert _pump(lambda: not w._controller.is_active)
    assert w._run_dark_available(), "expected a resolved dark master plan"

    # Fresh camera: no DB configured → §4 Create root path.
    w._bpm_root = None
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: str(new_base)),
    )
    # Auto-accept the §5 map-creation offer (and the §4 Create is entered
    # directly via _choose_bpm_database below).
    _fake_message_box(monkeypatch, "Create Bad Pixel Map")

    kinds = []
    real_start = w._start_operation
    def recording_start(snapshot):
        kinds.append(snapshot.kind)
        return real_start(snapshot)
    w._start_operation = recording_start

    logs = []
    real_log = w._log
    def recording_log(text):
        logs.append(text)
        return real_log(text)
    w._log = recording_log

    try:
        w._choose_bpm_database(None, str(tmp_path / "out"), create=True)
        # Wait for the full serialized chain to drain.
        assert _pump(lambda: not w._controller.is_active, timeout_ms=60000)

        # save → create map → export, in order, with no dropped start.
        assert kinds == ["save_bpm_settings", "create_bpm_map", "export"], kinds
        assert not any("operation is already running" in t for t in logs), logs

        # A real promoted revision was created in the new base.
        from zecalibrator.bpm.store import load_bad_pixel_database

        loaded = load_bad_pixel_database(new_base)
        assert loaded.state == "OPENED", loaded.reason_code
        promoted = [r for r in loaded.database.revisions() if r.state == "promoted"]
        assert promoted, "expected a promoted revision after map creation"
        assert promoted[-1].detector_k == 30.0
    finally:
        _close(w)
