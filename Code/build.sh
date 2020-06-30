#!/bin/bash

echo "-- Build Process --"
nvcc CUDA_MatrixMult.cu
g++ -o seq.out seq_Multi.cpp
g++ opencl_MatrixMult.cpp -L/usr/local/cuda-10.2/targets/x86_64-linux/lib/ -lOpenCL -o ocl.out
g++ omp_Multi.cpp -fopenmp -O3 -o parallel.out

echo "initTime,execTime,retrieveTime" > results_seq.csv
echo "initTime,execTime,retrieveTime" > results_omp.csv
echo "initTime,cpyTime,execTime,retrieveTime" > results_cuda.csv
echo "initTime,cpyTime,execTime,retrieveTime" > results_ocl.csv

runs=50

for i in $(seq 1 ${runs})
do
  echo "-- Run ${i}/${runs} --"
  echo "- CUDA"
  ./a.out >> results_cuda.csv
  echo "- Seq"
  ./seq.out >> results_seq.csv
  echo "- OpenCL"
  ./ocl.out >> results_ocl.csv
  echo "- OpenMP"
  ./parallel.out >> results_omp.csv
done

echo "-- Debug Run --"

./a.out 1
./seq.out 1
./ocl.out 1
./parallel.out 1

