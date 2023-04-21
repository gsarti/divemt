.PHONY: quality style

# Check that source code meets quality standards

quality:
	black --diff --check --config pyproject.toml ./
	ruff  --no-fix --config pyproject.toml ./

# Format source code automatically

style:
	black --config pyproject.toml ./
	ruff --config pyproject.toml ./

# Setup the library

setup:
	pip install requirements.txt
	pip install -e .

# Preprocess all the data in the data/raw folder

preprocess:
	python scripts/preprocess.py  --output_texts --add_edits --add_evals --output_merged


render_reports:
	for merged_file in data/processed/merged/*.tsv; do
		Rscript -e "rmarkdown::render('notebooks/postediting_effort.Rmd', output_file='reports/index.html', params=list(datapath="${merged_file}"))"
	done
	