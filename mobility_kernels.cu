/* Raul P. Pelaez 2021. Callers for the kernels in mobility_kernels.cuh.
   The funtions in this file take CPU arrays and, after uploading them to the GPU, call the GPU mobility kernels. Finally returning the results in the CPU.

 */
#include<tuple>
#include<thrust/device_vector.h>
#include"allocator.h"
#include <stdio.h>
#include"interface.h"
#include"mobility_kernels.cuh"
namespace mobility_cuda{
  using resource = device_memory_resource;

  using resource = mobility_cuda::device_memory_resource;
  using device_temporary_memory_resource = mobility_cuda::pool_memory_resource_adaptor<resource>;
  template<class T> using allocator_thrust = mobility_cuda::polymorphic_allocator<T, device_temporary_memory_resource, thrust::cuda::pointer<T>>;
  template<class T>  using cached_vector = thrust::device_vector<T, allocator_thrust<T>>;

  std::tuple<int,int> set_number_of_threads_and_blocks(int num_elements){
    // This functions uses a heuristic method to determine
    // the number of blocks and threads per block to be
    // used in CUDA kernels.
    int threads_per_block=512;
    if((num_elements/threads_per_block) < 512)
      threads_per_block = 256;
    if((num_elements/threads_per_block) < 256)
      threads_per_block = 128;
    if((num_elements/threads_per_block) < 128)
      threads_per_block = 64;
    if((num_elements/threads_per_block) < 128)
      threads_per_block = 32;
    int num_blocks = (num_elements-1)/threads_per_block + 1;
    return std::make_tuple(threads_per_block, num_blocks);
  }

  void single_wall_mobility_trans_times_force_cuda(const real* h_pos, const real* h_forces, real *h_u, real eta, real a, real Lx, real Ly, real Lz, int number_of_blobs){
    //Determine number of threads and blocks for the GPU
    int threads_per_block, num_blocks;
    std::tie(threads_per_block, num_blocks) = set_number_of_threads_and_blocks(number_of_blobs);
    cached_vector<real> pos(h_pos, h_pos + 3*number_of_blobs);
    cached_vector<real> forces(h_forces, h_forces + 3*number_of_blobs);
    cached_vector<real> Mv(3 * number_of_blobs);
    //Compute mobility force product
    velocity_from_force<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(pos.data()),
							   thrust::raw_pointer_cast(forces.data()),
							   thrust::raw_pointer_cast(Mv.data()),
							   number_of_blobs,
							   eta, a,
							   Lx,Ly,Lz);
    //Copy data from GPU to CPU (device to host)
    thrust::copy(Mv.begin(), Mv.end(), h_u);
  }

}

