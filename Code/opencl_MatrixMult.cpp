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

int main(int argc, char* argv[]){

	size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

	// Allocate host memory
	float* host_A = (float*) malloc(size);
	float* host_B = (float*) malloc(size);
	float* host_C = (float*) malloc(size);

	// Allocate device memory
	cl_mem device_A;
	cl_mem device_B;
	cl_mem device_C;

	// Fill the host matrices
	srand(42);

	for(int i = 0; i < size; i++){
		host_A[i] = (((float) rand()) * 0.7) % 10; 
		host_B[i] = (((float) rand()) * 0.7) % 10; 
	}


	cl_unit device_cnt = 0;
	clGetPlatformIDs(0, 0, &device_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(device_cnt, platform_ids, NULL);

	// Connect to compute device
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

	if (err != CL_SUCCESS) {
		sdt::cout << "Failed to create a device group!" << std::endl;
		return EXIT_FAILURE;
	}

	// Create a compute context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) {
		std::cout << "Failed to create a compute context!" << std::endl;
		return EXIT_FAILURE;
	}

	// Create a command queue
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands) {
		std::cout << "Failed to create a command queue!" << std::endl;
		return EXIT_FAILURE;
	}


	// Create the compute program from source file
	char* KernelSource;
	long lFileSize;

	lFileSize = LoadOpenCLKernel("matrixmul_kernel.cl", &KernelSource, false);
	if (lFileSize < 0L) {
		std::cout << "File reading failed!" << std::endl;
		return 1;
	}




//-----------------------
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

