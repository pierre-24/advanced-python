install:
	pip install -e .[dev]

flake8:
	flake8 neighbour_search tests

doctest:
	python -m doctest -v neighbour_search/*.py

mypy:
	mypy neighbour_search/*.py tests/*.py

lint: flake8 doctest mypy

test:
	pytest tests

doc:
	sphinx-build docs _build/