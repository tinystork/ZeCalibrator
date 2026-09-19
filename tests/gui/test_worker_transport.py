"""Architectural regression: the worker↔GUI transport boundary.

The worker carries Python payloads across the thread boundary through shared
``queue.Queue`` objects (plain Python references), never through Qt
``Signal(object)`` marshalling. This test reads the ACTUAL Qt signal signature
of ``_OperationWorker`` via its ``QMetaObject`` — the same metadata Qt uses to
marshal a queued cross-thread connection — and fails if any signal ever gains an
``object``-carrying parameter.

The introspection is structural, not a source-text grep: it enumerates the
``QMetaMethod`` entries Qt registered for the class and reads each signal's
declared parameter types. A future ``Signal(object)`` or ``Signal(str, object)``
on ``_OperationWorker`` therefore trips the object-carrying assertion below
(verified by a positive control), and any renamed/added/removed signal trips the
exact-surface assertions.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore

from zecalibrator.gui.worker import _OperationWorker

# Built-in QObject signals that are not part of the worker's transport surface.
_QOBJECT_BUILTIN_SIGNALS = {b"destroyed", b"objectNameChanged"}

# The exact transport surface, normalized to Python type labels. ``start_op``,
# ``relay`` and ``ended`` are no-payload wakeups; ``started`` and ``failed`` are
# typed scalar notifications only.
_EXPECTED_SIGNATURES = {
    "start_op": (),
    "started": ("str",),
    "relay": (),
    "failed": ("str", "str", "str"),
    "ended": (),
}

# Qt parameter-type names that indicate a marshalled Python/opaque object. The
# lower-case match is deliberate: it covers ``PyObject`` (what ``Signal(object)``
# declares) as well as ``QVariant``/``QObject`` variants.
_OBJECT_MARKERS = ("object", "variant")

_QT_TYPE_ALIASES = {
    "QString": "str",
    "QByteArray": "bytes",
    "int": "int",
    "double": "float",
    "bool": "bool",
}


def _qt_to_python(qt_type: str) -> str:
    """Map a Qt parameter-type name to the Python label used in assertions."""
    return _QT_TYPE_ALIASES.get(qt_type, qt_type)


def _is_object_carrying(qt_type: str) -> bool:
    """True if a Qt parameter type marshals a Python/opaque object."""
    lowered = qt_type.lower()
    return any(marker in lowered for marker in _OBJECT_MARKERS)


def _worker_signal_signatures() -> dict[str, tuple[str, ...]]:
    """Read the declared Qt signature of every worker transport signal.

    Uses ``QMetaObject``/``QMetaMethod`` rather than the ``Signal`` descriptor
    (which in PySide6 does not expose its declared argument types) or grepping
    source text. Returns ``{signal_name: tuple[qt_type, ...]}``.
    """
    meta = _OperationWorker.staticMetaObject
    signatures: dict[str, tuple[str, ...]] = {}
    for idx in range(meta.methodCount()):
        method = meta.method(idx)
        if method.methodType() != QtCore.QMetaMethod.MethodType.Signal:
            continue
        name = bytes(method.name())
        if name in _QOBJECT_BUILTIN_SIGNALS:
            continue
        signatures[name.decode()] = tuple(
            bytes(pt).decode() for pt in method.parameterTypes()
        )
    return signatures


def test_worker_signal_surface_is_exact():
    """The worker exposes exactly the five transport signals — no more, no less."""
    signatures = _worker_signal_signatures()
    assert set(signatures) == set(_EXPECTED_SIGNATURES), (
        f"unexpected worker signal surface: {sorted(signatures)}"
    )


def test_worker_signal_parameter_types_are_exact():
    """Each signal declares the exact expected scalar/none payload."""
    signatures = _worker_signal_signatures()
    for name, expected in _EXPECTED_SIGNATURES.items():
        actual = tuple(_qt_to_python(t) for t in signatures[name])
        assert actual == expected, (
            f"signal {name!r} declared {actual!r}, expected {expected!r}"
        )


def test_no_worker_signal_carries_an_object_payload():
    """No signal may marshal a Python/opaque object across the thread boundary."""
    signatures = _worker_signal_signatures()
    offenders = {
        name: types
        for name, types in signatures.items()
        if any(_is_object_carrying(t) for t in types)
    }
    assert not offenders, (
        f"worker signals carrying object payloads (forbidden transport): {offenders}"
    )


def test_object_carrying_detection_is_not_vacuous():
    """Positive control: the detector must actually flag ``Signal(object)``.

    Proves the invariant test is structurally meaningful and cannot silently
    pass while an object-carrying signal exists.
    """

    class _Probe(QtCore.QObject):
        payload = QtCore.Signal(object)
        mixed = QtCore.Signal(str, object)
        scalar = QtCore.Signal(str)

    meta = _Probe.staticMetaObject
    flagged: dict[str, tuple[str, ...]] = {}
    for idx in range(meta.methodCount()):
        method = meta.method(idx)
        if method.methodType() != QtCore.QMetaMethod.MethodType.Signal:
            continue
        name = bytes(method.name())
        if name in _QOBJECT_BUILTIN_SIGNALS:
            continue
        types = tuple(bytes(pt).decode() for pt in method.parameterTypes())
        if any(_is_object_carrying(t) for t in types):
            flagged[name.decode()] = types

    assert set(flagged) == {"payload", "mixed"}
    assert "scalar" not in flagged
