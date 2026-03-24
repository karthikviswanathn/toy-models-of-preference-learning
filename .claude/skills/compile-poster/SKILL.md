---
name: compile-poster
description: Compile the LaTeX poster to PDF, keeping aux files in poster/build/
disable-model-invocation: true
allowed-tools: Bash(*)
---

# Compile Poster

Compile the LaTeX poster to PDF. Auxiliary files go into `poster/build/` to keep the poster directory clean.

## Steps

1. Create `poster/build/` if needed
2. Copy `.tex` and `.sty` into `build/`
3. Symlink `figs/` and `logos/` into `build/` so LaTeX can find them
4. Run `pdflatex` in `build/`
5. Copy resulting PDF back to `poster/poster.pdf`

```bash
cd /pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning/poster

PDFLATEX=/scratch/project_465002390/texlive/.TinyTeX/bin/x86_64-linux/pdflatex

mkdir -p build
cp poster.tex beamerthemeposter.sty build/
ln -sfn "$(pwd)/figs" build/figs
ln -sfn "$(pwd)/logos" build/logos

cd build && \
$PDFLATEX -interaction=nonstopmode poster.tex && \
$PDFLATEX -interaction=nonstopmode poster.tex && \
cp poster.pdf ../poster.pdf && \
echo "Poster compiled: poster/poster.pdf"
```