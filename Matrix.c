//
// Created by joeyv on 6/3/2023.
//
#include <malloc.h>
#include "Matrix.h"
double randDouble(double max, double min);
Matrix* initMatrix (int rows, int columns){
    Matrix *m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->columns = columns;
    m->values = malloc(sizeof(double*)*rows);
    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        m->values[rowIndex] = malloc(sizeof(double)*columns);
        for (int colIndex = 0; colIndex < columns; colIndex++) {
            m->values[rowIndex][colIndex] = 0;
        }
    }
    return m;
}
void randomizeMatrix(Matrix* m, double maxValue, double minValue){
    for (int rowIndex = 0; rowIndex < m->rows; rowIndex++) {
        for (int colIndex = 0; colIndex < m->columns; colIndex++) {
            m->values[rowIndex][colIndex] = randDouble(maxValue, minValue);
        }
    }
}
void printMatrix(Matrix *m){
    for (int rowIndex = 0; rowIndex < m->rows; rowIndex++) {
        for (int colIndex = 0; colIndex < m->columns; colIndex++) {
            printf("%f ", m->values[rowIndex][colIndex]);
        }
        printf("\n");
    }
}

double randDouble(double max, double min) {
    double value = (double)rand()/RAND_MAX;
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

    for(int rowIndex=0; rowIndex < rows; rowIndex++) {
        for(int columnIndex=0; columnIndex < cols; columnIndex++) {
            out->values[rowIndex][columnIndex] = a->values[rowIndex][columnIndex] + b->values[rowIndex][columnIndex];
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

    for(int rowIndex=0; rowIndex < rows; rowIndex++) {
        for(int columnIndex=0; columnIndex < cols; columnIndex++) {
            out->values[rowIndex][columnIndex] = a->values[rowIndex][columnIndex] * b->values[rowIndex][columnIndex];
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

    for(int i=0; i < rows; i++) {
        for(int j=0; j < cols; j++) {
            double temp_total = 0;
            for(int k=0; k < shared; k++) {
                temp_total += a->values[i][k] * b->values[k][j];
            }
            out->values[i][j] = temp_total;
        }
    }
}

void inPlaceScaleMatrix(Matrix *matrix, double n) {
    int rows = matrix->rows;
    int cols = matrix->columns;

    for(int i=0; i < rows; i++) {
        for(int j=0; j < cols; j++) {
            matrix->values[i][j] *= n;
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

    for(int rowIndex=0; rowIndex < rows; rowIndex++) {
        for(int columnIndex=0; columnIndex < cols; columnIndex++) {
            destination->values[rowIndex][columnIndex] = original->values[rowIndex][columnIndex];
        }
    }
}

Matrix* getColumn(Matrix* m, int index) {
    assert(index < m->columns);
    assert(index >=0);

    Matrix *column = initMatrix(m->rows,1);
    for(int i = 0; i < m->rows; i++) {
        column->values[i][0] = m->values[i][index];
    }
    return column;
}
Matrix* getRow(Matrix* m, int index) {
    assert(index < m->rows);
    assert(index >= 0);

    Matrix *column = initMatrix(1,m->columns);
    for(int i = 0; i < m->columns; i++) {
        column->values[0][i] = m->values[index][i];
    }
    return column;
}

void deleteMatrix(Matrix *m) {
    for (int rowIndex = 0; rowIndex < m->rows; rowIndex++) {
        free( m->values[rowIndex]);
    }
    free(m->values);
    free(m);
}

void applyFunction(Matrix *matrix, double (*function)(double val)) {
    int rows = matrix->rows;
    int cols = matrix->columns;

    for(int i=0; i < rows; i++) {
        for(int j=0; j < cols; j++) {
            matrix->values[i][j] = function(matrix->values[i][j]);
        }
    }
}
