#define MATRIX_SIZE 1024

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.hpp>
#include <chrono>


char* readKernelFile(const char* filename, long* _size) {
	FILE* file = fopen(filename, "r");
	if (!file) {
		std::cout << "Failed to open kernel file" << std::endl;
		exit(1);
	}

	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	rewind(file);

	char* source = (char*) malloc((size+1) * sizeof(char));
	fread(source, 1, size * sizeof(char), file);
	source[size] = '\0';
	fclose(file);

	*_size = (size+1);
	return source;
}


int main(int argc, char* argv[]){

	cl_int err;
	cl_context context = 0;
	cl_device_id device_id = 0;
	cl_command_queue commands = 0;
	cl_program program = NULL;
	cl_kernel kernel = NULL;

	size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
	int dim_size = MATRIX_SIZE;


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
	cl_mem device_A;
	cl_mem device_B;
	cl_mem device_C;


	cl_uint device_cnt = 0;
	clGetPlatformIDs(0, 0, &device_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(device_cnt, platform_ids, NULL);

	// Connect to compute device
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

	if (err != CL_SUCCESS) {
		std::cout << "Failed to create a device group!" << std::endl;
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

	KernelSource = readKernelFile("kernel.cl", &lFileSize);
	if (lFileSize < 0L) {
		std::cout << "File reading failed!" << std::endl;
		return 1;
	}

	program = clCreateProgramWithSource(context, 1, (const char**) &KernelSource, NULL, &err);
	if (!program) {
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
	kernel = clCreateKernel(program, "matrixMul", &err);
	if (!kernel || err != CL_SUCCESS) {
		std::cout << "Failed to create compute kernel!" << std::endl;
		return EXIT_FAILURE;
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration_alloc = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	//  ------------------- Memory Copy -------------------
	start = std::chrono::high_resolution_clock::now();
	// Move to device memory 
	device_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, host_A, &err);
	device_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, host_B, &err);
	device_C = clCreateBuffer(context, CL_MEM_READ_WRITE, size, host_C, &err);

	if (!device_A || !device_B || !device_C) {
		std::cout << "Failed to allocate device memory" << std::endl;
		return EXIT_FAILURE;
	}


	end = std::chrono::high_resolution_clock::now();
	auto duration_memcpy = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	//  ------------------- Calculation -------------------
	start = std::chrono::high_resolution_clock::now();
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
	
	localWorkSize[0] = 32;
	localWorkSize[1] = 32;
	globalWorkSize[0] = 1024;
	globalWorkSize[1] = 1024;

	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		std::cout << "Failed to execute kernel! " << err << std::endl;
		return EXIT_FAILURE;
	}

	clFinish(commands); // wait for the kernel to finish
	end = std::chrono::high_resolution_clock::now();
	auto duration_calc = std::chrono::duration_cast<std::chrono::microseconds>(end - start);


	//  ------------------- Recopy Data -------------------
	start = std::chrono::high_resolution_clock::now();
	// Retrieve result from device
	err = clEnqueueReadBuffer(commands, device_C, CL_TRUE, 0, size, host_C, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		std::cout << "Failed to read output array! " << err << std::endl;
		return EXIT_FAILURE;
	}
	
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

