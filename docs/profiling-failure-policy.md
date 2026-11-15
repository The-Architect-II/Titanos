# Profiling failure and completed results

An explicit worker preserves the exact completed backend result if profiler exit or event collection raises `MemoryError`. Requested profiling remains visible in the assignment, but `profiling_evidence` reports failure and `observed_gpu_indices` is unavailable. This does not manufacture audio or GPU execution evidence.

The child accepts no further request after a profiling failure. The parent uses its existing bounded forced-stop path and keeps that explicit worker unavailable until it is recreated. Retirement failures remain visible in worker metadata; a completed result is still returned. No request is replayed and profiling is not silently disabled for another inference.

A genuine inference error or stop/control exception remains primary if profiler exit also fails. Non-memory profiling/CUDA errors propagate rather than becoming successful results. The failed collector reference is released before marshalling a completed result. Sanitized worker logs retain exception type, phase and code locations; they omit exception values and request/configuration data. This policy does not change model settings, memory caps or the generic process-stop implementation.
