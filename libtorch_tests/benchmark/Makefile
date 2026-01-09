CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1
INCLUDE_DIRS = -I../libtorch/include -I../libtorch/include/torch/csrc/api/include
LIB_DIRS = -L../libtorch/lib -Wl,-rpath,../libtorch/lib
LIBS = -ltorch -ltorch_cpu -lc10

all: libtorch_fun_tests cuda_bench_all_3d

libtorch_fun_tests: libtorch_fun_tests.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $< $(LIB_DIRS) $(LIBS)

cuda_bench_all_3d: cuda_bench_all_3d.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $< $(LIB_DIRS) -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10

clean:
	rm -f libtorch_fun_tests cuda_bench_all_3d
