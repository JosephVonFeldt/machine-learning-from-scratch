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
void broadcastAdd(Matrix *a, Matrix *b, Matrix *out);
void broadcastMultiply(Matrix *a, Matrix *b, Matrix *out);
void matMultiply(Matrix *a, Matrix *b, Matrix *out);
void inPlaceScaleMatrix(Matrix *matrix, double n);
#endif //HELLOWORLD_MATRIX_H
