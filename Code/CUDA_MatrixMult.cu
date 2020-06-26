#define MATRIX_SIZE 4096

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <chrono>


// Kernel Function Definition
// Pass the matrices as arrays
__global__ void SqMatrixMul(float* A, float* B, float* C, int N) {

	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;
	float cell_sum = 0.0;

	// In case the Number of threads does not match the matrix size
	// some threads will skip the work
	if (ROW < N && COL < N) {
		for(int i = 0; i < N; i++){
			cell_sum += A[ROW * N + i] * B[i * N + COL]; 
		}
		C[ROW * N + COL] = cell_sum;
	}
}

int main(int argc, char* argv[]){

	size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

	// Allocate host memory
	float* host_A = (float*) malloc(size);
	float* host_B = (float*) malloc(size);
	float* host_C = (float*) malloc(size);

	// Fill the host matrices
	srand(42);

	for(int i = 0; i < (int) (size / sizeof(float)); i++){
		host_A[i] = fmod(((float) rand()) * 0.7, 10.0); 
		host_B[i] = fmod(((float) rand()) * 0.7, 10.0); 
	}

	//  ------------------- Allocation -------------------
	auto start = std::chrono::high_resolution_clock::now();

	// Allocate device memory
	float* device_A;
	cudaMalloc(&device_A, size);
	float* device_B;
	cudaMalloc(&device_B, size);
	float* device_C;
	cudaMalloc(&device_C, size);

	// Declare the number of blocks per grid and the number of threads per block
	// Only 1024 threads per block are allowed -> 32 * 32
	// The dimensions for blocks in grids are maximum (2^31-1, 65535)
	
	dim3 threadsPerBlock(MATRIX_SIZE, MATRIX_SIZE);
	dim3 blocksPerGrid(1,1);

	if (size > 1024 * sizeof(float)){
		threadsPerBlock.x = 32;
		threadsPerBlock.y = 32;
		int blocks = ceil(double(MATRIX_SIZE)/double(32)); //=32
		blocksPerGrid.x = blocks;
		blocksPerGrid.y = blocks;
	}
	 
	auto end = std::chrono::high_resolution_clock::now();
	auto duration_alloc = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	//  ------------------- Memory Copy -------------------
	start = std::chrono::high_resolution_clock::now();
	// Move the data to the devcie memory
	cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

	end = std::chrono::high_resolution_clock::now();
	auto duration_memcpy = std::chrono::duration_cast<std::chrono::microseconds>(end - start);


	//  ------------------- Calculation -------------------
	start = std::chrono::high_resolution_clock::now();
	// invoke kernel
	SqMatrixMul<<<blocksPerGrid,threadsPerBlock>>>(device_A, device_B, device_C, MATRIX_SIZE);

	end = std::chrono::high_resolution_clock::now();
	auto duration_calc = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	
	//  ------------------- Recopy Data -------------------
	start = std::chrono::high_resolution_clock::now();
	// Copy the results back to the host
	cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

	// Free decive memory
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);

	end = std::chrono::high_resolution_clock::now();
	auto duration_free = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << duration_alloc.count() << "," << duration_memcpy.count() << "," << duration_calc.count() << "," << duration_free.count() << std::endl;


	if (argc > 1) {
		std::cerr << "A[0][0]:" << host_A[0] << " , B[0][0]:" << host_B[0] << " ,C[0][0]:" << host_C[0] << std::endl;
		std::cerr << "A[0][453]:" << host_A[453] << " , B[0][521]:" << host_B[521] << " ,C[0][1000]:" << host_C[1000] << std::endl;
	}

	// Free host memory
	free(host_A);
	free(host_B);
	free(host_C);
}

