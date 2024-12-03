## DRY Kernels

Experimentation with the DRY sampler in CUDA. The average implementation is slow when batched, so this is an attempt at parallelizing it.


```
mkdir build && cd build
cmake ..
make
```

You will need the CUDA toolkit and Torch installed.
