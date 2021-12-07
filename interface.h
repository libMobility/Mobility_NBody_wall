/* Raul P. Pelaez 2021. Declarations for the CUDA kernel callers in mobility_kernels.cu.
   The purpose of this header is to allow for separated compilation of GPU and CPU code.
 */
#ifndef MOBILITY_CUDA_INTERFACE_H
#define MOBILITY_CUDA_INTERFACE_H
namespace mobility_cuda{
#ifndef DOUBLE_PRECISION
  using real = float;
#else
  using real = double;
#endif

  void single_wall_mobility_trans_times_force_cuda(const real* h_pos,const real* h_forces, real *h_u,
						   real eta, real a,
						   real Lx, real Ly, real Lz,
						   int number_of_blobs);
}
#endif
