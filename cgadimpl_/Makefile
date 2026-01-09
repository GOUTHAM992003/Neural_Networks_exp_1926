# --- Master Makefile for the Entire Project ---
# Orchestrates build of both TensorLib and cgadimpl

.PHONY: all clean rebuild run-snippet run-snippet-force build-tensor build-cgadimpl force-rebuild-tensor force-rebuild-cgadimpl

# Default: Build everything incrementally
all: build-tensor build-cgadimpl

# Step 1: Build TensorLib (The Backend) - Incremental
build-tensor:
	@echo "=== Building TensorLib ===" 
	$(MAKE) -C tensor

# Step 2: Build cgadimpl (The Frontend) - Incremental
build-cgadimpl: build-tensor
	@echo "=== Building cgadimpl ==="
	$(MAKE) -C cgadimpl

# Force rebuild TensorLib (clean first)
force-rebuild-tensor:
	@echo "=== FORCE Rebuilding TensorLib ==="
	$(MAKE) -C tensor clean
	$(MAKE) -C tensor

# Force rebuild cgadimpl (clean first)
force-rebuild-cgadimpl: force-rebuild-tensor
	@echo "=== FORCE Rebuilding cgadimpl ==="
	$(MAKE) -C cgadimpl clean
	$(MAKE) -C cgadimpl

# Clean everything
clean:
	@echo "=== Cleaning TensorLib ==="
	$(MAKE) -C tensor clean
	@echo "=== Cleaning cgadimpl ==="
	$(MAKE) -C cgadimpl clean

# Rebuild (Clean + Build) - same as force
rebuild: clean all

# ============================================================
# run-snippet: INCREMENTAL build (only rebuilds changed files)
# Usage: make run-snippet FILE=regression_own.cpp
# ============================================================
run-snippet: all
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify a file."; \
		exit 1; \
	fi
	@echo "=== Running Snippet: $(FILE) ==="
	$(MAKE) -C cgadimpl run-snippet FILE=$(abspath $(FILE))

# ============================================================
# run-snippet-force: FORCE rebuild both libraries first
# Usage: make run-snippet-force FILE=regression_own.cpp
# ============================================================
run-snippet-force: force-rebuild-cgadimpl
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify a file."; \
		exit 1; \
	fi
	@echo "=== Running Snippet (After Force Rebuild): $(FILE) ==="
	$(MAKE) -C cgadimpl run-snippet FILE=$(abspath $(FILE))
