# Plotting

Plotting code currently exists as a Jupyter notebook in this folder.
- `plotting.ipynb` — generates all figures from the saved RDS / NPZ outputs in `results/`.

The notebook reads from `results/runs/...`, so run the corresponding R / Python sampling scripts first (see [REPRODUCIBILITY.md §2](../../REPRODUCIBILITY.md) for the paper-section ↔ notebook-figure mapping).

## ⚠ Known issue: filename casing on Linux

The file is tracked in git as **`Plotting.ipynb`** (capital `P`), but the working-tree filename on the maintainer's macOS system is `plotting.ipynb` (lowercase). macOS / APFS is case-insensitive by default, so both names resolve to the same file there.

On Linux / case-sensitive filesystems, `git checkout` will produce `Plotting.ipynb`, and any reference to the lowercase `plotting.ipynb` will fail. Workaround until this is renamed in the repo:

```bash
# Linux/CI users only
ln -s Plotting.ipynb plotting.ipynb
# or
git mv Plotting.ipynb plotting.ipynb   # then commit
```

The README and `REPRODUCIBILITY.md` reference the lowercase form, which is the intended canonical name.
