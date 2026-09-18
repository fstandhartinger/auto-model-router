"""Trusted runner executed *inside* the sandbox to grade one coding answer.

It is a separate file on purpose. The answer under test may read it - it holds
no secret. The one thing the answer must not be able to produce is the per-run
nonce, and that arrives on **stdin**, is consumed before the answer executes,
and is never written to the filesystem.

Exit codes let the grader say *why* something failed without trusting anything
the answer printed:

* 11 - the answer raised while it was being loaded
* 12 - a hidden test assertion failed
* 13 - a hidden test raised something other than AssertionError

Success is the line ``VERDICT <nonce>`` on stdout, written only after every
assertion has passed.

This file is never imported by the harness; it is copied into the sandbox and
run there.
"""

import sys
import traceback


def main() -> int:
    nonce = sys.stdin.readline().strip()
    if not nonce:
        return 10

    namespace = {"__name__": "__submission__"}
    try:
        with open("solution.py") as handle:
            source = handle.read()
        exec(compile(source, "solution.py", "exec"), namespace)   # noqa: S102
    except BaseException:                                          # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return 11

    scope = dict(namespace)
    try:
        with open("checks.py") as handle:
            exec(compile(handle.read(), "checks.py", "exec"), scope)   # noqa: S102
    except AssertionError:
        traceback.print_exc(file=sys.stderr)
        return 12
    except BaseException:                                          # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return 13

    sys.stdout.write("\nVERDICT " + nonce + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
