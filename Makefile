.PHONY: \
	all \
	format \
	format-python \
	lint \
	lint-python

all: format-python lint-python

format: format-python

format-python: python/vplot.py 
	black $^

lint: lint-python

lint-python: python/vplot.py 
	pylint $^
	flake8 $^
