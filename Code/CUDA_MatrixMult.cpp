#define MATRIX_SIZE 500




int main(){

	size_t size = MATRIX_SIZE * MATRIX_SIZE * size(float);

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






	// Free decive memory
	cudaFree(device_A);
	cudaFree(device_B);
	cudaFree(device_C);

	// Free host memory
	free(host_A);
	free(host_B);
	free(host_C);
}

