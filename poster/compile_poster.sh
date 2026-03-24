#!/bin/bash
# Compile the LaTeX poster, keeping aux files in build/
set -e

cd "$(dirname "$0")"

PDFLATEX=/scratch/project_465002390/texlive/.TinyTeX/bin/x86_64-linux/pdflatex

mkdir -p build
cp poster.tex beamerthemeposter.sty build/
ln -sfn "$(pwd)/figs" build/figs
ln -sfn "$(pwd)/logos" build/logos

cd build
$PDFLATEX -interaction=nonstopmode poster.tex
$PDFLATEX -interaction=nonstopmode poster.tex
cp poster.pdf ../poster.pdf

echo "Poster compiled: poster/poster.pdf"
