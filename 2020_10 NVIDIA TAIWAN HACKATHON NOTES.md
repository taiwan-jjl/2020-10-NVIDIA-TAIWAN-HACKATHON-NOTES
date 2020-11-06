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

  * ### CPU version with Intel MKL

    VSL_BRNG_MT19937 (driver.c)  
    (A Mersenne Twister pseudorandom number generator.)

    VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 (driver.c)  
    (<https://software.intel.com/content/www/us/en/develop/documentation/mkl-vsnotes/top/testing-of-distribution-random-number-generators/continuous-distribution-random-number-generators/gaussian-vsl-rng-method-gaussian-boxmuller2.html>)

    `#define BRNG    VSL_BRNG_MT19937` (header.h)

    error handling (utils.c)

    Ref:  
    mkl-2018-developer-reference-c_0.pdf

  * ### GPU version with NVIDIA cuRAND

    `CURAND_RNG_PSEUDO_MT19937`  
    is the pseudo random number generator corresponding to the MKL one we use. However, the GPU one does not offer the same API format and parameters which are available in MKL. In general, it is still possible to implement the GPU version with the same PRNG algorithm and some coding work.

    Ref:  
    CURAND_Library.pdf, 2.1. Generator Types.

  * ### The difficulty in parallel and distributed PRN generation
  
    In order to accelerate the PRN generation further more, multi-GPU and multi-node method is a must. However, it runs into an issue which is the "offset" parameter.

    Ref:  
    CURAND_Library.pdf

    2.2.2. Offset  
    The offset parameter is used to skip ahead in the sequence. If offset = 100, the first random number generated will be the 100th in the sequence. This allows multiple runs of the same program to continue generating results from the same sequence without overlap. Note that the skip ahead function is not available for the CURAND_RNG_PSEUDO_MTGP32 and CURAND_RNG_PSEUDO_MT19937 generators.

    It means that it is not doable to use the same seed and different offset value in different GPUs and nodes. It is a mathematical question that whether it is allowed to use the same "offset=0" and different seeds to generate PRN in the one simulation. It also causes the reproducibility issue. At this moment, I have not studied this topic enough to answer it.

    There is a possible solution which is to use another PRNG which offset parameter is available.

  * ### PRNG quality test
  
    There is a chapter "Testing" in cuRAND document which gives a statistic quality test on the PRNGs. It is a good one and not offered in MKL document. It is worth reading.

  * ### MKL API modification after Ver. 2018
  
    It is confirmed that DPLBE CPU version program cannot be compiled with MKL version later than 2018. It is because the API format is changed.  
    For the record, I do not consider keeping using the old MKL is a good solution but only as a workaround.

* ## NVTAGS

  The concept is correct and a better approach is to address this issue when the main algorithm is implemented. Use NVTAGS to assist and gain a better node topology. It is not practical to limit the program only could run in a certain node topology because it is not available most of the time in real life .  
  Also, the workload manager such as "slurm", the cluster hardware and topology are a part of this issue. It means that it cannot be addressed only by user level privilege but also administrator privilege.

* ## MPI choice

  Open MPI is officially supported by NVIDIA and the default library in "NVIDIA HPC SDK". However, if only the MPI performance itself is considered, especially not related to GPU, Open MPI is usually not the best one. There are many CPU-only clusters choose other MPI library like MVAPICH as their default.  
  Another very important issue which most people do not pay enough attention to is to compile and install the driver and middle-ware, not just the MPI library itself, correctly. Whether the RDMA, UCX, and other components are compiled and installed correctly, it would play an important role for MPI library to reach its full potential.  
  Sadly, even NCHC could not handle this satisfying to me.

* ## NCCL
  
  It is a subset of the MPI standard but for GPUs. It only provides limited APIs and mostly focuses on the collective communication. Use it if you could find an API which fits your needs best.

* ## The main programming language for NVIDIA GPU in the future

  In this session, I would like to talk about the paradigm shift of programming language for NVIDIA GPU and its impact to us. As an end-user and developer, I find out that it is unstoppable and irreversible and it is us who have to adapt and accept this changes. It has its own reasons in the industry although it may not be needed in the academia community.  

  At the beginning stage which I consider when there were only CUDA and PTX, the main languages in CUDA framework were C and FORTRAN. They were used to do the GPU programming directly and produce the libraries for further use. C and FORTRAN were equally supported and documents were written in both languages, especially, the sample codes. PTX plays a role as the fundamental instruction sets to the NVIDIA GPU devices and usually is used when the programmers really need to the control the device behavior at the lowest level.  

  Then it came to the stage when there were directive programming model such as OpenACC and C++ template library such as Thrust, and the C++ language were supported by the NVIDIA compiler. At that time, CUDA was still the framework added on the ISO C and C++ language, not as a part of the language standard. Also, it showed in the documents and development of NVIDIA software ecosystem that more and more C++ was used as the main or even sole language. FORTRAN was still supported but mostly used in the fundamental and mathematical libraries. The biggest feature of this stage is that there are functionalities which could be used by the programmers only in C++ language. For example, `nvcuda::wmma` namespace first introduced in CUDA 9.0 2017 is a part of C++ language and not overlapped with C language. It is the only way for the programmers to use the tensor cores except the PTX and libraries. It means that C++ now is the only high level language which could use the full functionalities of NVIDIA GPU. It has a tremendous impact to the C language users and even a tocsin to the FORTRAN language users.

  The third stage which I consider it started at 2020 began with the official support to the ISO C++ standard such as parallel algorithm and standard library since C++17. FORTRAN co-arrays and do-concurrent since FORTRAN 2018 are supported as well. The impact of the development is now programmers could use ISO C++ language standard alone, not ISO C++ and CUDA framework which uses kernel functions to offload to the GPU. Without CUDA framework, now the program could be more abstract, more general, more suitable for the heterogeneous computing. With the same source code, now it can be compiled and implemented into SIMD, multi-CPU, many-CPU, multi-GPU, multi-node, and even FPGA. It brings higher universality into the heterogeneous computing. Although it may not give the best performance when it specifically runs on the NVIDIA GPU compared to the same algorithm implemented in CUDA framework, the universality is still very attractive to the developers. Many modern FORTRAN features are also supported this time. However, the NVIDIA FORTRAN compiler is still buggy and cannot use the full functionalities as I mentioned above. I would say the future of FORTRAN for NVIDIA devices is not as promising as it looks. As for C language, due to its own language style, I would say it would be in the position between FORTRAN and C++, which is ambiguous in NVIDIA road map in the future.  

  In the traditional science community, C and FORTRAN are still the main programming language and there are many many legacy codes still in operation. The scientists in new generations are still trained in these two languages as well. It will cause wider and wider gaps between the science community and the latest technology. In the long term, how to deal with the legacy codes; upgrade or remake; how to upgrade the training to the new generations are the issues the science community must face one day.

* ## TENSOR CORE PROGRAMMING MODELS
  
  * DEEP LEARNING FRAMEWORKS
  * CUDA LIBRARIES
  * CUDA WMMA
  * CUTLASS

* ## NVSHMEM

  * ### Traditional MPI VS. NVSHMEM

  ![Traditional MPI VS. NVSHMEM](https://github.com/taiwan-jjl/2020-10-NVIDIA-TAIWAN-HACKATHON-NOTES/blob/main/pic/mpi-nvshmem-explainer-diagram.svg?raw=true)
  ![Memory Model](https://github.com/taiwan-jjl/2020-10-NVIDIA-TAIWAN-HACKATHON-NOTES/blob/main/pic/mem_model.png?raw=true)

  * ### Requirements

    * volta or newer
    * CUDA 10.1 or newer
    * and other dependencies

  * ### NCHC bashrc

    ```bashrc
    module purge
    module load compiler/gnu/7.3.0
    module load nvhpc/20.7
    ```

  * ### Proof of Concept

    ![demo](https://github.com/taiwan-jjl/2020-10-NVIDIA-TAIWAN-HACKATHON-NOTES/blob/main/pic/ring.png?raw=true)

    ```C
    #include <stdio.h>  //standard C header file
    #include <cuda.h>  //CUDA header file
    #include <nvshmem.h>  //nvshmem header file
    #include <nvshmemx.h>  //nvshmem header file

    //device kernel function
    __global__ void simple_shift(int *destination) {
        int mype = nvshmem_my_pe();  //from
        int npes = nvshmem_n_pes();  //total devices
        int peer = (mype + 1) % npes;  //to

        nvshmem_int_p(destination, mype, peer);  //from mype, put one integer mype, to peer at address destination
    }

    __global__ void check(int *destination) {  //check in device
        printf("%p: pe%d destination pointer\n", destination, nvshmem_my_pe());
        printf("%d: pe%d destination value\n", *destination, nvshmem_my_pe());
    }

    int main(void) {
        int mype_node, msg;
        cudaStream_t stream;  //Declares a stream handle

        nvshmem_init();  //initialize nvshmem
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);  //return the index within the node
        //NVSHMEMX_TEAM_NODE is a predefined named constant handle
        cudaSetDevice(mype_node);  //monopolize one device for each thread
        cudaStreamCreate(&stream);  //Allocates a stream

        int *destination = (int *) nvshmem_malloc(sizeof(int));  //nvshmem memory operation

        check<<<1, 1, 0, stream>>>(destination);  //verify nvshmem operation

        for(int i=1;i<=1000000;i++)  //just make it run longer
        {

        simple_shift<<<1, 1, 0, stream>>>(destination);  //NVIDIA GPU device kernel function
        //kernel<<<number of blocks, number of threads per block, number of bytes in shared memory, associated stream>>> ; Execution Configuration
        nvshmemx_barrier_all_on_stream(stream);  //synchronizing all PEs on stream at once
        cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream);
        //on stream, asynchronous copy with respect to the host, from destination in device, to msg on host, sizeof(int)

        cudaStreamSynchronize(stream);  //Blocks until stream has completed all operations

        }

        check<<<1, 1, 0, stream>>>(destination);  //verify nvshmem operation

        printf("%d: received message %d\n", nvshmem_my_pe(), msg);

        nvshmem_free(destination);  //nvshmem memory operation
        nvshmem_finalize();  //finalize nvshmem
        return 0;
    }
    ```
  
  * ### compile

    `nvcc -rdc=true -ccbin gcc -gencode=arch=compute_70,code=sm_70 -I /opt/ohpc/twcc/hpc_sdk/Linux_x86_64/20.7/comm_libs/nvshmem/include ./jjl_nvshmem_demo.cu -o ./jjl_nvshmem_demo -L /opt/ohpc/twcc/hpc_sdk/Linux_x86_64/20.7/comm_libs/nvshmem/lib -lnvshmem -lcuda`

  * ### run

    `~/opt/bin/nvshmrun -n 8 -ppn 8 ./jjl_nvshmem_demo`

    Due to technical issues, it is easier to build your own nvshmrun and run the demo.
