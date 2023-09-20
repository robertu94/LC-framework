# LC-framework

LC is a framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs

## Building with cmake

Requires cmake@3.18 or newer, Python, and a compiler that supports OpenMP.  Only supports the CPU version for now.

```
cmake -B builddir -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp
cmake --build builddir
cmake --build builddir --target install
```

+ `CMAKE_BUILD_TYPE` -- `Release` uses optimization, where as `Debug` uses debug symbols
+ `CMAKE_INSTALL_PREFIX` -- where to install the compiled artifacts
