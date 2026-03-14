# Show environment info
info:
    uv run python -c "import torch; print('CUDA available: ' + str(torch.cuda.is_available()))"
    nvcc --version

# ── Test ──────────────────────────────────────────────────────────────────────

# Run the full test suite (parallel, no benchmarks)
test *args:
    uv run pytest -n auto --dist=loadfile --benchmark-disable {{ args }}

# Run tests with coverage report
coverage *args:
    uv run pytest -n auto --dist=loadfile --benchmark-disable \
        --cov=sources --cov-report=term-missing --cov-report=html {{ args }}

# ── Benchmark ─────────────────────────────────────────────────────────────────

# Run both assignment benchmarks and refresh assets/benchmarks/
bench: bench-assignment bench-batched

# Single-problem assignment benchmark (all solvers × all datasets)
bench-assignment:
    uv run python -c "from unitrack.assignment.lapjv._extension import load_extension; load_extension()"
    uv run pytest tests/unitrack/assignment/test_assignment_benchmark.py -s -v -n0 --benchmark-disable --timeout=600

# Batched assignment benchmark
bench-batched:
    uv run python -c "from unitrack.assignment.lapjv._extension import load_extension; load_extension()"
    uv run pytest tests/unitrack/assignment/test_assignment_batched_benchmark.py -s -v -n0 --benchmark-disable --timeout=600

# Batched benchmark with lapx comparison (requires: uv sync --extra bench)
bench-batched-full:
    uv run python -c "from unitrack.assignment.lapjv._extension import load_extension; load_extension()"
    uv run pytest tests/unitrack/assignment/test_assignment_batched_benchmark.py -s -v -n0 --benchmark-disable --timeout=600 --lapx

# ── Docs ──────────────────────────────────────────────────────────────────────

# Build the docyard documentation site
docs:
    nix run .#docs-build

# ── Housekeeping ──────────────────────────────────────────────────────────────

# Remove caches and build artefacts
clean:
    rm -rf dist build .pytest_cache .coverage .ruff_cache
    find . \( \
        -name "__pycache__" -o -name "*.egg-info" \
        -o -name ".pytest_cache" -o -name ".hypothesis" \
    \) -type d -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
