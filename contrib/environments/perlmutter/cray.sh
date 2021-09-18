module load PrgEnv-cray
module load cmake

export PATH=/global/common/software/m2878/shasta2105/upcxx/TESTING-PrgEnv-cray/bin:$PATH

export UCX_TLS=dc
export UCX_DC_MLX5_NUM_DCI=16

module list
which cc
which CC
which g++
which gcc
which nvcc
which upcxx

export MHM2_CMAKE_EXTRAS="-DCMAKE_C_COMPILER=$(which cc) -DCMAKE_CXX_COMPILER=$(which CC) -DCMAKE_CUDA_COMPILER=$(which nvcc)"
