## Mobility solvers in CUDA

Several mobility evaluators, adapted from the pycuda code written by Florencio Balboa, are available.  
I just took these kernels from the RigidMultiblob repository and placed them in a separated, C++ only, repository.  
A lot of mobility solvers are available, all of them using an N^2 sum and optionally incorporating PBC by summing over images of the system.  
Kernels for computing contributions from torque and/or force are available  

mobility_kernels.cu implements a series of CPU callers to the kernels defined in mobility_kernels.cuh.   
These callers accept CPU arrays, taking care of uploading to/downloading from the GPU and calling the CUDA kernels.  

The file interface.h contains declarations for the functions in mobility_kernels.cu to ease separate compilation of CPU and GPU code.  

