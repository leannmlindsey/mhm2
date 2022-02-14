module load PrgEnv-gnu
#module load PrgEnv-cray
#module load PrgEnv-nvidia
#module load nvidia
module load gcc/9.3.0
module load cmake
module load cuda
module load cudatoolkit

#export PATH=/global/common/software/m2878/shasta2105/upcxx/TESTING-PrgEnv-nvidia/bin:$PATH
#export PATH=/global/common/software/m2878/shasta2105/upcxx/TESTING-PrgEnv-cray/bin:$PATH
#export PATH=/global/common/software/m2878/shasta2105/upcxx/TESTING-PrgEnv-gnu/bin:$PATH
export PATH=$SCRATCH/install-upcxx-2021.3-PrgEnv-gnu-9.3.0/bin:$PATH

module list
which cc
which CC
which g++
which gcc
which nvcc
which upcxx

export MHM2_CMAKE_EXTRAS="-DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_CUDA_COMPILER=$(which nvcc)"
#export MHM2_CMAKE_EXTRAS="-DCMAKE_C_COMPILER=/opt/cray/pe/craype/2.7.9/bin/cc -DCMAKE_CXX_COMPILER=/opt/cray/pe/craype/2.7.9/bin/CC -DCMAKE_CUDA_COMPILER=$(which nvcc)"
