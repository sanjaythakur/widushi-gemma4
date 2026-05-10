"""Hardware abstraction layer (mic, camera, display).

All hardware sources are async and behind narrow interfaces so the
orchestrator never blocks on USB I/O. Phase 2 ships pure stubs that
yield silence / blank frames — actual capture lands in later phases.
"""
