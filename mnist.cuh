//
// Created by joeyv on 6/4/2023.
//
#ifdef __cplusplus
extern "C" {
#endif
#ifndef NEURALNETS_MNIST_CUH
#define NEURALNETS_MNIST_CUH

#include "MatrixGPU.cuh"

//#include "Matrix.h"
int lineCount(char *fileName); // This probably isn't secure
void getMnistFileData(Matrix *input, Matrix *output, char *filename);


#endif //NEURALNETS_MNIST_CUH
#ifdef __cplusplus
}
#endif