"""GUI package.

PySide6 imports are isolated to the Qt modules (``worker``, ``window``) and to
lazy imports inside ``app``/``identity``. Importing this package performs no Qt
import: the pure helpers (``service``, ``settings``, ``presentation``) and the
``app`` module are safe to import headlessly, while ``worker``/``window`` import
PySide6 at module scope and are only loaded by the Qt entry point or Qt tests.
"""
