#!/bin/bash

echo "-- Build Process --"
nvcc CUDA_MatrixMult.cu
g++ -o seq.out seq_Multi.cpp
g++ opencl_MatrixMult.cpp -L/usr/local/cuda-10.2/targets/x86_64-linux/lib/ -lOpenCL -o ocl.out
g++ omp_Multi.cpp -fopenmp -O3 -o parallel.out

echo "initTime,execTime,retrieveTime" > results.csv

runs=50

for i in $(seq 1 ${runs})
do
  echo "-- Run ${i}/${runs} --"
  ./a.out >> results.csv
  ./seq.out >> results.csv
  ./ocl.out >> results.csv
  ./parallel.out >> results.csv
done

echo "-- Debug Run --"

./a.out 1
./seq.out 1
./ocl.out 1
./parallel.out 1

