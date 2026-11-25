"""Single-threaded BLAS for bit-exact reproduction.

Multi-threaded BLAS changes the summation order of matrix products (differences of ~1e-17),
which can swap exactly tied candidates. Every experiment entry point imports this module
before numpy. The number of threads is RPGR_BLAS_THREADS (default 1, the setting used to
verify all numbers of the paper)."""
import os

_threads = os.environ.get("RPGR_BLAS_THREADS", "1")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = _threads
