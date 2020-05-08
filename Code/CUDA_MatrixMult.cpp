#define MATRIX_SIZE 1024

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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

int main(){

	size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

	// Allocate host memory
	float* host_A = (float*) malloc(size);
	float* host_B = (float*) malloc(size);
	float* host_C = (float*) malloc(size);

	// Allocate device memory
	float* device_A;
	cudaMalloc(&device_A, size);
	float* device_B;
	cudaMalloc(&device_B, size);
	float* device_C;
	cudaMalloc(&device_C, size);

	// Fill the host matrices
	srand(42);

	for(int i = 0; i < size; i++){
		host_A[i] = (((float) rand()) * 0.7) % 10; 
		host_B[i] = (((float) rand()) * 0.7) % 10; 
	}

	// Move the data to the devcie memory
	cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);



	// Declare the number of blocks per grid and the number of threads per block
	// Only 1024 threads per block are allowed -> 32 * 32
	// The dimensions for blocks in grids are maximum (2^31-1, 65535)
	
	if (size > 1024 * sizeof(float)){
		dim3 threadsPerBlock(32, 32);
		int blocks = ceil(double(MATRIX_SIZE)/double(32));
		dim3 blocksPerGrid(blocks, blocks);
	} else {
		dim3 threadsPerBlock(MATRIX_SIZE, MATRIX_SIZE);
		dim3 blocksPerGrid(1,1);
	}
	// And invoke kernel
	SqMatrixMul<<<blocksPerGrid,threadsPerBlock>>>(device_A, device_B, device_C, MATRIX_SIZE);

	// Copy the results back to the host
	cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

	// Free decive memory
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);

	// Free host memory
	free(host_A);
	free(host_B);
	free(host_C);
}

