#!/bin/bash

nvcc CUDA_MatrixMult.cu

for i in $(seq 1 10)
do
  ./a.out
done

