//
// Created by joeyv on 6/3/2023.
//

#ifndef HELLOWORLD_MATRIX_H
#define HELLOWORLD_MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

typedef struct Matrix Matrix;
Matrix* initMatrix (int rows, int columns);
void randomizeMatrix(Matrix* m, double maxValue, double minValue);
void printMatrix(Matrix *m);
void deleteMatrix(Matrix *m);
struct Matrix{
    int rows;
    int columns;
    double** values;
};
#endif //HELLOWORLD_MATRIX_H
