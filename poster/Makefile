TEXDIR = /scratch/project_465002390/texlive/.TinyTeX/bin/x86_64-linux
PDFLATEX = $(TEXDIR)/pdflatex
LATEXMK = $(TEXDIR)/latexmk
PROJECT_ROOT = /pfs/lustrep3/projappl/project_465002390/fair_stuff/toy-models-of-preference-learning

.PHONY: all poster figs figs-cpu clean

all: figs poster

poster: poster.pdf

poster.pdf: poster.tex beamerthemeposter.sty
	$(PDFLATEX) -interaction=nonstopmode poster.tex
	$(PDFLATEX) -interaction=nonstopmode poster.tex

figs:
	cd $(PROJECT_ROOT) && sbatch run_job.sh poster/poster_figs.py --device cuda

figs-cpu:
	cd $(PROJECT_ROOT) && python poster/poster_figs.py --device cpu

clean:
	rm -f poster.pdf poster.aux poster.log poster.nav poster.snm poster.vrb poster.out poster.toc
