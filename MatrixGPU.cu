//
// Created by joeyv on 6/9/2023.
//
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
using namespace std;
#include "MatrixGPU.cuh"

#include <malloc.h>
double randDouble(double max, double min);
__global__ void mat_mul(double *mat1, double *mat2, double *outMat, int outRowCount, int sharedCount, int outColCount);
Matrix* initMatrix(int rows, int columns) {
    auto *m = new Matrix();
    m->rows = rows;
    m->columns = columns;
    m->values = static_cast<double *>(malloc(sizeof(double *) * rows * columns));
    //cudaMallocHost(&m->values, sizeof(double *) * rows * columns);
    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int colIndex = 0; colIndex < columns; colIndex++) {
            m->values[rowIndex * columns + colIndex] = 0;
        }
    }
    return m;
}
void randomizeMatrix(Matrix *m, double maxValue, double minValue) {
    for (int rowIndex = 0; rowIndex < m->rows; rowIndex++) {
        for (int colIndex = 0; colIndex < m->columns; colIndex++) {
            m->values[rowIndex * m->columns + colIndex] = randDouble(maxValue, minValue);
        }
    }
}
void printMatrix(Matrix *m) {
    for (int rowIndex = 0; rowIndex < m->rows; rowIndex++) {
        for (int colIndex = 0; colIndex < m->columns; colIndex++) {
            printf("%f ", m->values[rowIndex * m->columns + colIndex]);
        }
        printf("\n");
    }
}

double randDouble(double max, double min) {
    double value = (double) rand() / RAND_MAX;
    value = (max - min) * value + min;
    return value;
}

void broadcastAdd(Matrix *a, Matrix *b, Matrix *out) {
    // Checking rows
    assert(a->rows == b->rows);
    assert(b->rows == out->rows);
    int rows = a->rows;

    // Checking columns
    assert(a->columns == b->columns);
    assert(b->columns == out->columns);
    int cols = a->columns;

    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int columnIndex = 0; columnIndex < cols; columnIndex++) {
            out->values[rowIndex * out->columns + columnIndex] = a->values[rowIndex * a->columns + columnIndex] + b->values[rowIndex * b->columns + columnIndex];
        }
    }
}

void broadcastMultiply(Matrix *a, Matrix *b, Matrix *out) {
    // Checking rows
    assert(a->rows == b->rows);
    assert(b->rows == out->rows);
    int rows = a->rows;

    // Checking columns
    assert(a->columns == b->columns);
    assert(b->columns == out->columns);
    int cols = a->columns;

    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int columnIndex = 0; columnIndex < cols; columnIndex++) {
            out->values[rowIndex * out->columns + columnIndex] = a->values[rowIndex * a->columns + columnIndex] * b->values[rowIndex * b->columns + columnIndex];
        }
    }
}

void matMultiply(Matrix *a, Matrix *b, Matrix *out) {
    // Checking row/column compatibility
    assert(a->columns == b->rows);
    int shared = a->columns;
    assert(a->rows == out->rows);
    assert(b->columns == out->columns);
    int rows = out->rows;
    int cols = out->columns;

    if (0) {
        double *gpuMatA;
        double *gpuMatB;
        double *gpuMatOut;

        cudaMalloc((void**) &gpuMatA, sizeof(double)*a->rows*a->columns);
        cudaMalloc((void**) &gpuMatB, sizeof(double)*b->rows*b->columns);
        cudaMalloc((void**) &gpuMatOut, sizeof(double)*out->rows*out->columns);

        cudaMemcpy(gpuMatA, a->values, sizeof(double)*a->rows*a->columns, cudaMemcpyHostToDevice);
        cudaMemcpy(gpuMatB, b->values, sizeof(double)*b->rows*b->columns, cudaMemcpyHostToDevice);

        mat_mul<<<cols, rows>>>(gpuMatA, gpuMatB, gpuMatOut, rows, a->columns, cols);



        cudaMemcpy(out->values, gpuMatOut, sizeof(double)*out->rows*out->columns, cudaMemcpyDeviceToHost);

        cudaFree(gpuMatA);
        cudaFree(gpuMatB);
        cudaFree(gpuMatOut);

    }else {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double temp_total = 0;
                for (int k = 0; k < shared; k++) {
                    temp_total += a->values[i * a->columns + k] * b->values[k * b->columns + j];
                }
                out->values[i * out->columns + j] = temp_total;
            }
        }
    }
}

void inPlaceScaleMatrix(Matrix *matrix, double n) {
    int rows = matrix->rows;
    int cols = matrix->columns;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->values[i * matrix->columns + j] *= n;
        }
    }
}

void copyInto(Matrix *original, Matrix *destination) {
    // Checking rows
    assert(original->rows == destination->rows);
    int rows = original->rows;

    // Checking columns
    assert(original->columns == destination->columns);
    int cols = original->columns;

    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int columnIndex = 0; columnIndex < cols; columnIndex++) {
            destination->values[rowIndex * cols + columnIndex] = original->values[rowIndex * cols + columnIndex];
        }
    }
}

Matrix *getColumn(Matrix *m, int index) {
    assert(index < m->columns);
    assert(index >= 0);

    Matrix *column = initMatrix(m->rows, 1);
    for (int i = 0; i < m->rows; i++) {
        column->values[i] = m->values[i * m->columns + index];
    }
    return column;
}

Matrix *getRow(Matrix *m, int index) {
    assert(index < m->rows);
    assert(index >= 0);

    Matrix *column = initMatrix(1, m->columns);
    for (int i = 0; i < m->columns; i++) {
        column->values[i] = m->values[index * m->columns + i];
    }
    return column;
}

void getColumnArr(Matrix *m, int index, double out[]) {
    assert(index < m->columns);
    assert(index >= 0);
    for (int i = 0; i < m->rows; i++) {
        out[i] = m->values[i * m->columns + index];
    }
}

void getRowArr(Matrix *m, int index, double out[]) {
    assert(index < m->rows);
    assert(index >= 0);
    for (int i = 0; i < m->columns; i++) {
        out[i] = m->values[index * m->columns + i];
    }
}

void deleteMatrix(Matrix *m) {
    //cudaFreeHost(m->values);
    free(m->values);
    free(m);
}

void applyFunction(Matrix *matrix, double (*function)(double val)) {
    int rows = matrix->rows;
    int cols = matrix->columns;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->values[i * matrix->columns + j] = function(matrix->values[i * matrix->columns + j]);
        }
    }
}

double getValue(Matrix * m, int i, int j) {
    return m->values[i*m->columns + j];
}

void setValue(Matrix * m, int i, int j, double value) {
    m->values[i*m->columns + j] = value;
}

__global__ void mat_mul(double *mat1, double *mat2, double *outMat, int outRowCount, int sharedCount, int outColCount) {
    unsigned int num = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = num/outColCount;
    unsigned int col = num%outColCount;
    if (col < outColCount && row < outRowCount) {
        double  sum = 0;
        for (unsigned int i = 0; i < sharedCount; i++) {
            sum += mat1[row * sharedCount + i] * mat2[i * outColCount + col];
        }
        outMat[row * outColCount + col] = sum;
    }
}
