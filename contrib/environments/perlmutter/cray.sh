module load PrgEnv-cray
module load cmake
module load cuda
module load cudatoolkit

export PATH=/global/common/software/m2878/shasta2105/upcxx/TESTING-PrgEnv-cray/bin:$PATH

export UCX_TLS=dc
export UCX_DC_MLX5_NUM_DCI=16

export UCX_RC_MLX5_RETRY_COUNT=40
export UCX_UD_MLX5_TIMEOUT=600000000.00us
export UCX_RC_MLX5_TIMEOUT=1200000.00us
export UCX_DC_MLX5_TIMEOUT=1200000.00us
export UCX_DC_MLX5_RETRY_COUNT=40
export UCX_WARN_UNUSED_ENV_VARS=n

module list
which cc
which CC
which g++
which gcc
which nvcc
which upcxx

export MHM2_CMAKE_EXTRAS="-DCMAKE_C_COMPILER=$(which cc) -DCMAKE_CXX_COMPILER=$(which CC) -DCMAKE_CUDA_COMPILER=$(which nvcc)"
