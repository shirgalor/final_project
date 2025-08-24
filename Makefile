PYTHON ?= python

.PHONY: all build clean test

all: build

build:
	$(PYTHON) setup.py build_ext --inplace

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf build dist *.egg-info symnmf/__pycache__ tests/__pycache__ symnmf/_csymnmf.*.so symnmf/_csymnmf.*.pyd
