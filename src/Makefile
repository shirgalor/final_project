CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm
TARGET = symnmf
SOURCES = symnmf.c symnmf_tools.c
PYTHON ?= ./venv/bin/python

.PHONY: all clean build

all: $(TARGET)

# Build C executable with required flags
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

build: $(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf build dist *.egg-info src/__pycache__ tests/__pycache__
	rm -f src/*.so src/*.pyd