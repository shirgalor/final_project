# SymNMF Project (Python + optional C-accelerated core)

This repository contains a reference implementation of **Symmetric Nonnegative Matrix Factorization (SymNMF)**
for clustering via similarity graphs. It includes:

- Pure Python implementation (`symnmf/symnmf.py`) with NumPy.
- Optional C extension (`symnmf/symnmf.c`, `symnmf/symnmfmodule.c`, `symnmf/symnmf.h`) to accelerate the multiplicative updates.
- A simple analysis script (`analysis.py`) that generates a toy dataset, builds a similarity matrix, runs SymNMF,
  and applies k-means to rows of **H** to derive clusters.
- Tests in `tests/test_basic.py`.

## Quick start (Python-only)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python analysis.py --k 2 --dataset moons --plot
```

## Optional: Build the C extension
The Python-only version is sufficient. If you want extra speed, build the C extension:

```bash
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
# or
pip install .
```

This will compile the module `symnmf/_csymnmf.*` exposing a single function `update(H, W, max_iter, tol)`
that performs the SymNMF multiplicative updates in C.

> The Python implementation will automatically fall back if the C extension is not present.

## Project structure

```
symnmf_project/
├── analysis.py
├── requirements.txt
├── setup.py
├── Makefile
├── README.md
├── data/
│   └── example.csv            # (optional) placeholder for your data
├── symnmf/
│   ├── __init__.py
│   ├── symnmf.py              # Python implementation
│   ├── symnmf.h               # C header
│   ├── symnmf.c               # C core updates
│   └── symnmfmodule.c         # Python C API wrapper
└── tests/
    └── test_basic.py
```

## How it connects to k-means

1. Build a **similarity matrix** `W` from data `X` (e.g., Gaussian kernel or kNN).
2. Optionally normalize `W` (symmetric normalization).
3. Run **SymNMF** to factorize `W ≈ H Hᵀ`, with `H ≥ 0`, where `H` is `n × k`.
4. Run **k-means on the rows of `H`** to obtain final clusters.

## Notes

- The objective minimized is \(\|W - H H^T\|_F^2\) subject to \(H \ge 0\).
- Multiplicative updates: \(H \leftarrow H \odot \frac{WH}{H H^T H + \epsilon}\).
- Use `--tol` or `--max-iter` to control convergence.
- For reproducibility, pass `--seed`.
