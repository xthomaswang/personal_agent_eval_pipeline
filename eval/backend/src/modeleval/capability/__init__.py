"""The capability leg (DESIGN §6): harness epoch, runner, per-run checks and the allowlisted export.

Every module here except runner.main() is stdlib-only, so the orchestrator (Python 3.12) can import
them; runner.main() runs in envs/capability (Python 3.13) next to the vendored harness code.
"""
