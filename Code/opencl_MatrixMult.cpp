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
	int dim_size = MATRIX_SIZE;

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

	program = clCreateProgramWithSource(context, 1, (const char**) &KernelSource, NULL, &err);
	if (!proram) {
		std::cout << "Failed to create compute program!" << std::endl;
		return EXIT_FAILURE;
	}

	// Build the progam executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		std::cout << "Failed to build program!" << std::endl;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		std::cout << buffer << std::endl;
		return EXIT_FAILURE;
	}

	// Create the compute kernel in the program
	kernel = clCreateKernel(program, "matrixMul", &er);
	if (!kernel || err != CL_SUCCESS) {
		std::cout << "Failed to create compute kernel!" << std::endl;
		return EXIT_FAILURE;
	}

	// Allocate device memory 
	device_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, host_A, &err);
	device_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, host_B, &err);
	device_C = clCreateBuffer(context, CL_MEM_READ_WRITE, size, host_C, &err);

	if (!device_A || !device_B || !device_C) {
		std::cout << "Failed to allocate device memory" << std::endl;
		return EXIT_FAILURE;
	}

	// Launch OpenCL kernel
	// global: number of work-items executing the function
	// local: number of work-items making up a work-group
	size_t localWorkSize[2], globalWorkSize[2];

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &device_A);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &device_B);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &device_C);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *) &dim_size);

	if (err != CL_SUCCESS) {
		std::cout << "Failed to set kernel arguments!" << std::endl;
		return EXIT_FAILURE;
	}
	
	localWorkSize[0] = 16;
	localWorkSize[1] = 16;
	globalWorkSize[0] = 1024;
	globalWorkSize[1] = 1024;

	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		std::cout << "Failed to execute kernel! " << err << std::endl;
		return EXIT_FAILURE;
	}


	// Retrieve result from device
	err = clEnqueueReadBuffer(commands, device_C, CL_TRUE, 0, size, host_C, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		std::cout << "Failed to read output array! " << err << std::endl;
		return EXIT_FAILURE;
	}



	// Free host memory
	free(host_A);
	free(host_B);
	free(host_C);
}

