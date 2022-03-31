module load cpe-cuda
module load PrgEnv-nvidia
module load cmake
module load nvidia-nersc/21.5

export PATH=/global/common/software/m2878/shasta2105/upcxx/TESTING-PrgEnv-nvidia/bin:$PATH

module list
which cc
which CC
which g++
which gcc
which nvcc
which upcxx

export MHM2_CMAKE_EXTRAS="-DCMAKE_C_COMPILER=$(which cc) -DCMAKE_CXX_COMPILER=$(which CC) -DCMAKE_CUDA_COMPILER=$(which nvcc)"
