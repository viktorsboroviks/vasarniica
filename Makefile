.PHONY: \
	all \
	venv \
	format \
	format-python \
	lint \
	lint-python

all: format-python lint-python

PYTHON_VERSION := python3.12
venv: requirements.txt
	$(PYTHON_VERSION) -m venv venv
	./venv/bin/pip3 install --no-cache-dir --requirement requirements.txt

format: venv format-python

format-python: python/*.py
	. ./venv/bin/activate && \
		black $^

lint: lint-python lint-cpp

lint-cpp: \
		include/vasarniica/vtime.hpp
	cppcheck \
		--enable=warning,portability,performance \
		--enable=style,information \
		--enable=missingInclude \
		--inconclusive \
		--library=std,posix,gnu \
		--platform=unix64 \
		--language=c++ \
		--std=c++20 \
		--inline-suppr \
		--check-level=exhaustive \
		--suppress=missingIncludeSystem \
		--suppress=checkersReport \
		--checkers-report=cppcheck_report.txt \
		-I./include \
		$^

lint-python: venv \
		python/*.py
	. ./venv/bin/activate && \
		pylint $^ ; \
		flake8 $^

clean:

distclean: clean
	rm -rfv venv
