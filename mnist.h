//
// Created by joeyv on 6/4/2023.
//

#ifndef NEURALNETS_MNIST_H
#define NEURALNETS_MNIST_H
#include "MatrixGPU.cuh"
int lineCount(char* fileName); // This probably isn't secure
void getMnistFileData(Matrix* input, Matrix* output, char* filename);


#endif //NEURALNETS_MNIST_H
