# TestVulkanTrigPrecision

Test application for trigonometric function precision in Vulkan.

The application can be built using CMake on Windows (MSVC, MSYS2/MinGW) and Linux (GCC, Clang).
Precompiled binaries are provided in the [release section](https://github.com/chrismile/TestVulkanTrigPrecision/releases).


## Known Precision

NOTE: Precision can be either measured in absolute error or in ULP distance (units in the last place;
https://en.wikipedia.org/wiki/Unit_in_the_last_place).

Vulkan provides the following minimum 32-bit floating point precision. Individual ICDs can provide higher precision
(https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#spirvenv-op-prec).

| Function       | Precision | Scientific | Range         |
|----------------|-----------|------------|---------------|
| cos(x)         | $2^{-11}$ | 4.883e-4   | $[-\pi, \pi]$ |
| sin(x)         | $2^{-11}$ | 4.883e-4   | $[-\pi, \pi]$ |
| atan(x)        | 4096 ULP  |            | full          |
| inversesqrt(x) | 2 ULP     |            | full          |
| exp(x)         | 3+2ax ULP |            | full          |
| exp2(x)        | 3+2ax ULP |            | full          |
| log(x)         | $2^{-21}$ | 4.768e-7   | $[0.5, 2]$    |
| log(x)         | 3 ULP     |            | $R/[0.5, 2]$  |
| log2(x)        | $2^{-21}$ | 4.768e-7   | $[0.5, 2]$    |
| log2(x)        | 3 ULP     |            | $R/[0.5, 2]$  |

Some special cases:
- sqrt(x): Inherited from 1.0 / inversesqrt(x).
- tan(x): Inherited from sin(x)/cos(x)
- atan2(y,x): 4096 ULP
- asin(x): Inherited from atan2(x, sqrt(1-x*x))
- acos(x): Inherited from atan2(sqrt(1-x*x), x)


### NVIDIA

NVIDIA documents their 32-bit floating point precision for CUDA.
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__cosf#intrinsic-functions
  (https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#single-precision-only-intrinsic-functions)
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__cosf#standard-functions
  (https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#trigonometric-functions)

| Function   | Precision    | Scientific | Range         |
|------------|--------------|------------|---------------|
| __cosf(x)  | $2^{-21.41}$ | 3.589e-7   | $[-\pi, \pi]$ |
| __sinf(x)  | $2^{-21.19}$ | 4.180e-7   | $[-\pi, \pi]$ |
| __logf(x)  | $2^{-21.41}$ | 3.589e-7   | $[0.5, 2]$    |
| __logf(x)  | 2 ULP        |            | $R/[0.5, 2]$  |
| __log2f(x) | $2^{-22}$    | 2.384e-7   | $[0.5, 2]$    |
| __log2f(x) | 2 ULP        |            | $R/[0.5, 2]$  |
| cosf(x)    | 2 ULP        |            | full          |
| sinf(x)    | 2 ULP        |            | full          |
| tanf(x)    | 4 ULP        |            | full          |
| asinf(x)   | 2 ULP        |            | full          |
| acosf(x)   | 2 ULP        |            | full          |
| atanf(x)   | 2 ULP        |            | full          |
| expf(x)    | 2 ULP        |            | full          |
| exp2f(x)   | 2 ULP        |            | full          |
| logf(x)    | 1 ULP        |            | full          |
| log2f(x)   | 1 ULP        |            | full          |
| log10f(x)  | 2 ULP        |            | full          |
| sqrtf(x)   | 1 ULP (*)    |            | full          |
| rsqrtf(x)  | 2 ULP        |            | full          |
(*) Compute capability >= 5.2.

Some special cases:
- __tanf: Derived from __sinf(x) * (1 / __cosf(x))
- atan2f(y,x): 3 ULP
- __expf(x): <= 2 + floor(abs(1.173 * x)) ULP
- __exp10f(x): <= 2 + floor(abs(2.97 * x)) ULP

The Vulkan precision seems to match the intrinsic trigonometric functions. Below, results measured on an RTX 3090 can
be found.

| Function | Absolute | ULP error | Range         |
|----------|----------|-----------|---------------|
| cos(x)   | 4.172e-7 | 12303662  | $[-\pi, \pi]$ |
| sin(x)   | 3.576e-7 | 16777215  | $[-\pi, \pi]$ |
