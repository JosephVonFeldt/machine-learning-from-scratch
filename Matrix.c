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

void deleteMatrix(Matrix *m) {
    free(m->values);
    free(m);
}
