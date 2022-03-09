module use /gpfs/alpine/world-shared/csc296/summit/modulefiles

module rm PrgEnv-cray
module load PrgEnv-gnu

module load git
module load cmake

module rm xl
module load gcc
#module load cuda
#module load upcxx
export PATH=/gpfs/alpine/csc296/world-shared/crusher/upcxx/gnu/11.2.0/nightly/bin:$PATH
module list
which upcxx

#module load python
module load cray-python

#export GASNET_ODP_VERBOSE=0 # disable warnings
#export MHM2_CMAKE_EXTRAS="-DCMAKE_C_COMPILER=$(which mpicc) -DCMAKE_CXX_COMPILER=$(which mpicxx) -DCMAKE_CUDA_COMPILER=$(which nvcc)"
export MHM2_CMAKE_EXTRAS="-DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) -DENABLE_CUDA=Off"
export MHM2_BUILD_THREADS=8

# salloc/sbatch with: -A BIF115_crusher --ntasks-per-node=64 
