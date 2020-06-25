#define MATRIX_SIZE 4096

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <chrono>



int main(int argc, char* argv[]){

	size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

	// Allocate host memory
	float* A = (float*) malloc(size);
	float* B = (float*) malloc(size);
	float* C = (float*) malloc(size);


	// Fill the host matrices
	srand(42);

	for(int i = 0; i < (int) (size / sizeof(float)); i++){
		A[i] = fmod(((float) rand()) * 0.7, 10.0); 
		B[i] = fmod(((float) rand()) * 0.7, 10.0); 
	}


	float cell_sum;
	auto start = std::chrono::high_resolution_clock::now();
	// Multiply Matrix
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		for (int j = 0; j < MATRIX_SIZE; ++j) {
			cell_sum = 0.0;
			for (int k = 0; k < MATRIX_SIZE; ++k) {
				cell_sum += A[i * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + j];
			}
			C[i * MATRIX_SIZE + j] = cell_sum;
		}
	}


	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);

	

	std::cout << "0," << duration.count()<< ",0" << std::endl;
	if (argc > 1) {
		std::cerr << "A[0][0]:" << A[0] << " , B[0][0]:" << B[0] << " ,C[0][0]:" << C[0] << std::endl;
		std::cerr << "A[0][453]:" << A[453] << " , B[0][521]:" << B[521] << " ,C[0][1000]:" << C[1000] << std::endl;
	}


	// Free host memory
	free(A);
	free(B);
	free(C);
}

