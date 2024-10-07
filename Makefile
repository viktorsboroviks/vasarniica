.PHONY: \
	all \
	format \
	format-python \
	lint \
	lint-python

all: format-python lint-python

venv: requirements.txt
	python3 -m venv venv
	./venv/bin/pip3 install --no-cache-dir --requirement requirements.txt

format: venv format-python

format-python: python/*.py
	source ./venv/bin/activate ;\
		black $^

lint: venv lint-python

lint-python: python/*.py
	source ./venv/bin/activate ;\
		pylint $^ ;\
		flake8 $^
