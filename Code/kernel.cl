
//OpenCL Kernel
__kernel void matrixMul(__global float* A, __global float* B, __global float* C, int dimensions) {
	int tx = get_global_id(0);
	int ty = get_global_id(1);

	// value stores the element that is computed by the thread
	float value = 0;
	for (int k = 0; k < dimensions; ++k) {
		float elementA = A[ty * dimensions + k];
		float elementB = B[k * dimensions + tx];
		value += elementA * elementB;
	}

	// Write the matrix to device memory
	C[ty * dimensions + tx] = value;
}
