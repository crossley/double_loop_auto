CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
LDFLAGS ?=

BIN_DIR := bin
C_BIN := $(BIN_DIR)/model_spiking_cat_90vs180_c

.PHONY: all clean

all: $(C_BIN)

$(C_BIN): c/model_spiking_cat_90vs180.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) -lm

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)
