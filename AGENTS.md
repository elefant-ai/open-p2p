## Agent Notes: How to compile the Rust crate in this repo

In this project, the Rust library (`elefant_rust`) is built automatically via Python packaging using `maturin` when you execute Python with `uv`. Do not try to build the Rust crate directly with `maturin` or `cargo` for day‑to‑day workflows. Instead, run a Python entrypoint with `uv` and the build will happen automatically if needed.

### Recommended way

- Use `uv` to run any Python file that imports the `elefant` package. This will automatically trigger the Rust build as part of the package build step:

  - Example: run a simple import to force a build
    - `uv run python -c "import elefant; print('ok')"`

  - Example: run an existing script (also triggers build if needed)
    - `uv run python elefant/data/scripts/iter_dataset.py`

`uv` will resolve the Python environment and build the `elefant-rust` extension through `maturin` under the hood (PEP 517). You don't need to call `maturin` directly.

### Why not call maturin/cargo directly?

- Calling `maturin` directly may fail in CI/dev shells where the expected Python environment and `pyo3` configuration are not set up. The `uv run python ...` path correctly invokes the build backend with the environment established by `uv`.

### Forcing a rebuild

- If you changed Rust sources and want to ensure a rebuild, just re-run your Python entrypoint with `uv`:
  - `uv run python -c "import elefant; print('rebuilt')"`

`uv` will detect source changes and rebuild the native extension.

### Troubleshooting build errors

- Prefer reproducing by running a Python entrypoint with `uv` rather than invoking `maturin`/`cargo` manually.
- Common issues fixed recently:
  - Passing `Option<u64>` correctly from Python bindings:
    - Use `.map(|s| s * 1000)` to convert seconds to milliseconds instead of multiplying an `Option` directly.
  - Functions expecting `Option<u64>` should be given `Some(value)` instead of a bare integer.

If you still need low-level diagnostics, you can run (only for debugging):

- `cd elefant_rust && cargo check`

But for a working build and correct packaging, rely on `uv run python ...`.

# Agent Instructions

- After making any code changes, run `./devtools/scripts/lint.sh` from the repository root to ensure lint checks pass.
- If linting fails, fix the issues and rerun the script until it succeeds.
