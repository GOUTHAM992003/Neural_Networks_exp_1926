# Neural Networks Experiments

This repository contains experiments with neural networks using custom tensor and autograd libraries.

## ⚠️ Dependencies Not Included

The following dependencies are **NOT included** in this repository due to size:

### LibTorch (PyTorch C++ API)
- **Location:** `libtorch_tests/libtorch/`
- **Download:** https://pytorch.org/get-started/locally/
- **Size:** ~2.5 GB

To set up:
```bash
cd libtorch_tests/
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-*.zip
```

## Structure

```
test_env_gau/
├── cgadimpl_/           # Custom autograd implementation
├── Tensor_Implementations_kota/  # Custom tensor library
├── libtorch_tests/      # LibTorch benchmarks (libtorch not included)
├── pytorch_tests/       # PyTorch Python benchmarks
└── numpy_tests/         # NumPy benchmarks
```

## Building

```bash
cd cgadimpl_
make
make run-snippet FILE="regression_own_using_linear.cpp"
```

## Author
Goutham
