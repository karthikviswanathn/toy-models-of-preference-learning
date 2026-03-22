---
name: compile-pdf
description: Compile the LaTeX writeup to PDF, keeping aux files in build/
disable-model-invocation: true
allowed-tools: Bash(*)
argument-hint: "[filename.tex] (default: toy_models_preference_training.tex)"
---

# Compile PDF

Compile the LaTeX writeup to PDF. Auxiliary files go into `build/` to keep the directory clean.

## Steps

1. `cd` into `writeup/`
2. Run `pdflatex` then `bibtex` then `pdflatex` twice (to resolve citations and cross-references)
3. Copy the resulting PDF back to `writeup/`
4. Report success or failure

```bash
cd /pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning/writeup

FILE="${ARGUMENTS:-toy_models_preference_training.tex}"
mkdir -p build

pdflatex -interaction=nonstopmode -output-directory=build "$FILE" && \
BIBINPUTS=. bibtex build/"${FILE%.tex}" && \
pdflatex -interaction=nonstopmode -output-directory=build "$FILE" && \
pdflatex -interaction=nonstopmode -output-directory=build "$FILE" && \
cp "build/${FILE%.tex}.pdf" . && \
echo "PDF compiled: ${FILE%.tex}.pdf"
```
