#!/bin/bash

mhmxx_install_dir=$1

refs=arcticsynth-refs.fa
if [ ! -f "$refs" ]; then
    wget https://portal.nersc.gov/project/hipmer/MetaHipMer_datasets_12_2019/ArcticSynth/samples/${refs}.gz
    gunzip ${refs}.gz
fi
reads=arctic_sample_0.fq
if [ ! -f "$reads" ]; then
    wget https://portal.nersc.gov/project/hipmer/MetaHipMer_datasets_12_2019/ArcticSynth/samples/${reads}.gz
    gunzip ${reads}.gz
fi 

rm -rf /dev/shm/test-as
wd=`pwd`
$mhmxx_install_dir/bin/mhmxx.py -r $reads -o /dev/shm/test-as --checkpoint=no
$mhmxx_install_dir/bin/check_asm_quality.py --asm-dir /dev/shm/test-as --expected-quals $mhmxx_install_dir/share/good-arctic-sample0.txt --refs $refs
