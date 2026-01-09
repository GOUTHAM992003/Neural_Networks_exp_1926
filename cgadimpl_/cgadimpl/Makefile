# --- CGADIMPL Structured Makefile ---
# Matches TensorLib structure: lib/libcgadimpl.so, lib/objects/...

CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -fPIC -Iinclude -I../tensor/include
LDFLAGS_SHARED = -shared
LDFLAGS_BIN = -Llib -L../tensor/lib -lcgadimpl -ltensor -lcudart -fopenmp -L/usr/local/cuda/lib64
RPATH = -Wl,-rpath,$(PWD)/lib -Wl,-rpath,$(PWD)/../tensor/lib

# Directories
LIB_DIR = lib
OBJ_DIR = lib/objects

# Helper to find sources and map to objects
SRCS := $(shell find src -name "*.cpp")
OBJS := $(SRCS:src/%.cpp=$(OBJ_DIR)/src/%.o)

# Targets
TARGET_SO = $(LIB_DIR)/libcgadimpl.so
TARGET_A  = $(LIB_DIR)/libcgadimpl.a

.PHONY: all clean run-snippet directories

all: directories $(TARGET_SO) $(TARGET_A)

directories:
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(OBJ_DIR)

# Link Shared Library
$(TARGET_SO): $(OBJS)
	@echo "--- Linking Shared Library: $@ ---"
	$(CXX) $(LDFLAGS_SHARED) -o $@ $^

# Archive Static Library
$(TARGET_A): $(OBJS)
	@echo "--- Archiving Static Library: $@ ---"
	ar rcs $@ $^

# Compile Object Files (preserving directory structure)
$(OBJ_DIR)/src/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run Snippet
run-snippet: $(TARGET_SO)
	@if [ -z "$(FILE)" ]; then echo "Usage: make run-snippet FILE=..."; exit 1; fi
	@echo "--- Compiling Executable: $(FILE) ---"
	$(CXX) $(CXXFLAGS) $(FILE) -o snippet_runner $(LDFLAGS_BIN) $(RPATH)
	@echo "--- Running ---"
	LD_LIBRARY_PATH="./lib:../tensor/lib:$$LD_LIBRARY_PATH" ./snippet_runner

clean:
	rm -rf $(LIB_DIR) snippet_runner
