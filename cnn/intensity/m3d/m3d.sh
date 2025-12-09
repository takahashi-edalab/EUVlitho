#!/bin/sh
#$ -cwd
#$ -l node_o=1
#$ -l h_rt=0:10:00

export LD_LIBRARY_PATH="/gs/fs/tga-eda-takahashi/openblas/cuda/lib64:/gs/fs/tga-eda-takahashi/openblas/magma-2.9.0/lib:/gs/fs/tga-eda-takahashi/openblas/openblas/lib:$LD_LIBRARY_PATH"
export OPENBLAS_CORETYPE=ZEN5

./m3d.out