CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -lm
TARGET = symnmf/symnmf
SOURCES = symnmf/symnmf.c symnmf/symnmf_tools.c
PYTHON ?= ./venv/bin/python

.PHONY: all clean test build_ext

all: $(TARGET)

# Build C executable
$(TARGET): $(SOURCES)
	$(CC) -o $(TARGET) $(SOURCES) $(CFLAGS)

# Build Python extension
build_ext: venv
	$(PYTHON) setup.py build_ext --inplace

# Create virtual environment if it doesn't exist
venv:
	@if [ ! -d "venv" ]; then python3 -m venv venv && ./venv/bin/pip install setuptools numpy; fi

# Build both C executable and Python extension
build: $(TARGET) build_ext

clean:
	rm -f $(TARGET)
	rm -rf build dist *.egg-info symnmf/__pycache__ tests/__pycache__
	rm -f symnmf/*.so symnmf/*.pyd

test:
	python -m pytest -q
