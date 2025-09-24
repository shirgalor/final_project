CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm
TARGET = symnmf
SOURCES = symnmf.c symnmf_tools.c
PYTHON ?= ./venv/bin/python

.PHONY: all clean test build_ext

all: $(TARGET)

# Build C executable with required flags
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

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
	rm -rf build dist *.egg-info src/__pycache__ tests/__pycache__
	rm -f src/*.so src/*.pyd

test:
	python -m pytest -q
