.PHONY: \
	all \
	format \
	format-python \
	lint \
	lint-python

all: format-python lint-python

format: format-python

format-python: python/*.py
	black $^

lint: lint-python

lint-python: python/*.py
	pylint $^
	flake8 $^
