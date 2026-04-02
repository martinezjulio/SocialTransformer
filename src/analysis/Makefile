render:
	quarto render analysis.qmd

run:
	Rscript analysis.R

lint:
	Rscript -e 'lintr::lint_dir(".")'

fmt:
	Rscript -e 'styler::style_dir()'
