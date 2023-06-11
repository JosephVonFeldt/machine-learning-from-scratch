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
__global__ void apply_sigmoid(double *mat1, int len);
__global__ void mat_add(double *mat1, double *mat2, int len);
__global__ void mat_mul(double *mat1, double *mat2, double *outMat, int outRowCount, int sharedCount, int outColCount);


Matrix* initMatrix(int rows, int columns) {
    auto *m = new Matrix();
    m->rows = rows;
    m->columns = columns;
    double *temp;
    cudaMallocHost(&temp, sizeof(double *) * rows * columns);
    cudaMalloc(&m->values, sizeof(double *) * rows * columns);
    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int colIndex = 0; colIndex < columns; colIndex++) {
            temp[rowIndex * columns + colIndex] = 0;
        }
    }
    cudaMemcpy(m->values, temp, sizeof(double *) * rows * columns, cudaMemcpyHostToDevice);
    cudaFreeHost(temp);
    return m;
}
void randomizeMatrix(Matrix *m, double maxValue, double minValue) {
    double *temp;
    cudaMallocHost(&temp, sizeof(double *) * m-> rows * m->columns);
    for (int rowIndex = 0; rowIndex < m->rows; rowIndex++) {
        for (int colIndex = 0; colIndex < m->columns; colIndex++) {
            temp[rowIndex * m->columns + colIndex] = randDouble(maxValue, minValue);
        }
    }
    cudaMemcpy(m->values, temp, sizeof(double *) * m-> rows * m->columns, cudaMemcpyHostToDevice);
    cudaFreeHost(temp);
}
void printMatrix(Matrix *m) {
    double *temp;
    cudaMallocHost(&temp, m->rows * m->columns * sizeof(double));
    cudaMemcpy(temp, m->values, m->columns * m->rows * sizeof(double), cudaMemcpyDeviceToHost);
    for (int rowIndex = 0; rowIndex < m->rows; rowIndex++) {
        for (int colIndex = 0; colIndex < m->columns; colIndex++) {
            printf("%f ", temp[rowIndex * m->columns + colIndex]);
        }
        printf("\n");
    }
    cudaFreeHost(temp);
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
    mat_add<<<cols*rows,1>>>(a->values, b->values,  cols*rows);
    cudaDeviceSynchronize();
//    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
//        printf("\nrow %d \n", rowIndex);
//        for (int columnIndex = 0; columnIndex < cols; columnIndex++) {
//            printf("col %d\t", columnIndex);
//            double d = a->values[rowIndex * a->columns + columnIndex] + b->values[rowIndex * b->columns + columnIndex];
//            out->values[rowIndex * out->columns + columnIndex] = 1;
//            //a->values[rowIndex * a->columns + columnIndex] + b->values[rowIndex * b->columns + columnIndex];
//        }
//    }
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

    if (1) {
        mat_mul<<<cols, rows>>>(a->values, b->values, out->values, rows, a->columns, cols);
        cudaDeviceSynchronize();
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

    cudaMemcpy(destination->values,original->values,rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
//    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
//        for (int columnIndex = 0; columnIndex < cols; columnIndex++) {
//            destination->values[rowIndex * cols + columnIndex] = original->values[rowIndex * cols + columnIndex];
//        }
//    }
}

__global__ void getCol(double* mat, double* col, int colIndex, int numRows, int numCols){
    unsigned int num = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int curRow = num/numCols;
    unsigned int curCol = num%numCols;
    if (curRow < numRows && curCol == colIndex) {
        col[curRow] = mat[curRow * numCols + curCol];
    }
}
Matrix *getColumn(Matrix *m, int index) {
    assert(index < m->columns);
    assert(index >= 0);

    Matrix *column = initMatrix(m->rows, 1);
    getCol<<<m->columns * m->rows, 1>>>(m->values, column->values, index, m->rows, m->columns);
    cudaDeviceSynchronize();
    return column;
}
__global__ void getRow(double* mat, double* row, int rowIndex, int numRows, int numCols){
    unsigned int num = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int curRow = num/numCols;
    unsigned int curCol = num%numCols;
    if (curRow == rowIndex && curCol < numCols) {
        row[curCol] += mat[curRow * numCols + curCol];
    }
}
Matrix *getRow(Matrix *m, int index) {
    assert(index < m->rows);
    assert(index >= 0);

    Matrix *row = initMatrix(1, m->columns);
    getRow<<<m->columns * m->rows, 1>>>(m->values, row->values, index, m->rows, m->columns);
    cudaDeviceSynchronize();
    return row;
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
    cudaFree(m->values);
    free(m);
}

void applyFunction(Matrix *matrix, double (*function)(double val)) {
    int rows = matrix->rows;
    int cols = matrix->columns;
    apply_sigmoid<<<rows * cols, 1 >>>(matrix->values, rows*cols );
    //cudaDeviceSynchronize();
//    sync();
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            matrix->values[i * matrix->columns + j] = function(matrix->values[i * matrix->columns + j]);
//        }
//    }

}


double getValue(Matrix * m, int i, int j) {
    int ind = i * m->columns + j;
    double x;
    cudaMemcpy(&x, m->values + ind , sizeof(double), cudaMemcpyDeviceToHost);
    return x;
}

void setValue(Matrix * m, int i, int j, double value) {
    int ind = i * m->columns + j;
    double x = value;
    cudaMemcpy(m->values + ind * sizeof(double), &x, 1, cudaMemcpyHostToDevice);
}
void setValues(Matrix* m, double *values) {
    cudaMemcpy(m->values, values, sizeof(double) * m->columns * m->rows, cudaMemcpyHostToDevice);
}

void sync() {
    cudaDeviceSynchronize();
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
__global__ void mat_add(double *mat1, double *mat2,  int len) {
    unsigned int num = blockDim.x * blockIdx.x + threadIdx.x;
    if (num < len) {
        mat2[num] += mat1[num];
    }
}

__global__ void apply_sigmoid(double *mat1, int len) {
    unsigned int num = blockDim.x * blockIdx.x + threadIdx.x;
    if (num < len) {
        double x = mat1[num];
        x = 1 / (1 + exp(-x));
        mat1[num] = x;
    }
}
