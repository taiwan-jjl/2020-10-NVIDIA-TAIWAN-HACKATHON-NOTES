# 2020/10 NVIDIA TAIWAN HACKATHON NOTES

* ## Pinned memory and Unified memory

  * "Pinned(page-locked) memory" means it is "unpageable" in host memory, different from GPU device memory. It can be accessed by "direct memory access(DMA)" which gives a better IO performance. However, it can only be accessed in the host.
  * "Unified memory" means it is "pageable" and in a virtual unified memory space owned by the host and GPU device. It can be accessed by both CPU and GPU and the memory(variable/data) transfer is handled automatically.
  * They are two different ideas and can not be used simultaneously so far.

* ## NVIDIA MAGNUM IO

  "NVIDIA MAGNUM IO" is a full stack standard which includes from hardware, OS, driver to software. So far, it is still in pre-release stage and only be implemented in-house and with limited partners. There is no partners chosen in Taiwan, not even NCHC. Sadly, the end users are still not able to test and get its benefits.  
  However, IO, storage, and communication are also very important in scientific HPC, not just computation. I believe we must follow MAGNUM IO closely and adapt the applications for it in order to have a further improvement.

* ## GPU kernel function limitations

  Unlike using OpenMP standard, which basically has no additional limitations to the chosen high-level language, CUDA kernel functions have some limitations which is different from the chosen language and need to be borne in mind.

  Ref:
  * <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications>
  * <https://docs.nvidia.com/cuda/archive/10.1/pdf/CUDA_C_Programming_Guide.pdf>

  In a programmer's perspective, it could be memorized as follows:
  * There is a limitation on the total instructions in a kernel function. It means that the kernel function can not be too long.
  * There is a limitation on the total registers and memories in a thread/SM/etc.. It means that the performance of the operations you want to run will drop significantly because it has to use the global memory which is the slowest memory in the architecture if it exceeds the limitation. In another word, it means the algorithm is not suitable for the specific GPU architecture.

* ## cuRAND

  * ### CPU version with Intel MKL:

    VSL_BRNG_MT19937 (driver.c)  
    (A Mersenne Twister pseudorandom number generator.)

    VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 (driver.c)  
    (<https://software.intel.com/content/www/us/en/develop/documentation/mkl-vsnotes/top/testing-of-distribution-random-number-generators/continuous-distribution-random-number-generators/gaussian-vsl-rng-method-gaussian-boxmuller2.html>)

    `#define BRNG    VSL_BRNG_MT19937` (header.h)

    error handling (utils.c)

    Ref:  
    mkl-2018-developer-reference-c_0.pdf

  * ### GPU version with NVIDIA cuRAND:

    `CURAND_RNG_PSEUDO_MT19937`  
    is the pseudo random number generator corresponding to the MKL one we use. However, the GPU one does not offer the same API format and parameters which are available in MKL. In general, it is still possible to implement the GPU version with the same PRNG algorithm and some coding work.

    Ref:  
    CURAND_Library.pdf, 2.1. Generator Types.

  * ### The difficulty in parallel and distributed PRN generation:
  
    In order to accelerate the PRN generation further more, multi-GPU and multi-node method is a must. However, it runs into an issue which is the "offset" parameter.

    Ref:  
    CURAND_Library.pdf

    2.2.2. Offset  
    The offset parameter is used to skip ahead in the sequence. If offset = 100, the first random number generated will be the 100th in the sequence. This allows multiple runs of the same program to continue generating results from the same sequence without overlap. Note that the skip ahead function is not available for the CURAND_RNG_PSEUDO_MTGP32 and CURAND_RNG_PSEUDO_MT19937 generators.

    It means that it is not doable to use the same seed and different offset value in different GPUs and nodes. It is a mathematical question that whether it is allowed to use the same "offset=0" and different seeds to generate PRN in the one simulation. It also causes the reproducibility issue. At this moment, I have not studied this topic enough to answer it.

    There is a possible solution which is to use another PRNG which offset parameter is available.

  * ### PRNG quality test:
  
    There is a chapter "Testing" in cuRAND document which gives a statistic quality test on the PRNGs. It is a good one and not offered in MKL document. It is worth reading.

  * ### MKL API modification after Ver. 2018
  
    It is confirmed that DPLBE CPU version program cannot be compiled with MKL version later than 2018. It is because the API format is changed.  
    For the record, I do not consider keeping using the old MKL is a good solution but only as a workaround.

test1234