# Windows native command cleanup

Titanos runs native OCR commands in a private Windows Job. A timeout, cancellation
or output-budget error remains the primary failure even when cleanup also fails.
Cleanup failures are recorded as exception notes and `cleanup_errors` on the
retained Job object. The exception's `owned_job` reference lets an internal caller
inspect or explicitly recover that exact owned process and handle.

The Job handle becomes `None` only after `CloseHandle` succeeds. If cleanup fails
after an otherwise successful command, the request raises that cleanup failure.
The constructor attempts child termination, waiting and handle closure
independently, preserving the original launch or assignment error.

These rules report uncertain cleanup honestly; they do not guarantee that a
failing operating-system termination or close call eventually succeeds. There is
no automatic inference retry or broad process cleanup.

The Windows llama.cpp supervisor also owns its native executable and descendants
in a private Job. The non-inherited handle stays in the supervisor, so a forced
supervisor stop closes the handle and terminates the native child tree. Normal
stdin-EOF shutdown and native exit-code propagation retain their existing
interfaces. A Windows-only pipe watcher stops and joins when the native process
exits, avoiding a blocked buffered-stdin daemon during Python shutdown. The
POSIX supervisor path is unchanged.

Regression checks live in `tests/check_windows_job_cleanup.py` and
`tests/check_windows_job_caller.py`. `tests/check_tesseract.py` also exercises
harmless native Windows child processes for output limits, deadlines,
cancellation and descendant cleanup. Its OCR responses are synthetic; it does
not load a model or run real OCR.

`tests/check_native_tree_cleanup.py` is a Windows-only real process-lifecycle
check using sleeping Python child/grandchild fixtures. It checks forced
supervisor termination, stdin EOF and normal native exit, with an outer private
Job containing failures. It waits on retained process handles and does not run a
model, query a GPU or contact an API. This regression does not retrospectively
classify any earlier model-run failure or replace package/model qualification.
