//
// Created by joeyv on 6/9/2023.
//
#ifdef __cplusplus
extern "C" {
#endif
#ifndef NEURALNETS_MATRIXGPU_CUH
#define NEURALNETS_MATRIXGPU_CUH
    typedef struct Matrix Matrix;
    struct Matrix {
        int rows;
        int columns;
        double* values;
    };
    Matrix* initMatrix(int rows, int columns);
    void randomizeMatrix(Matrix* m, double maxValue, double minValue);
    void printMatrix(Matrix* m);
    void deleteMatrix(Matrix* m);
    void copyInto(Matrix* original, Matrix* destination);
    void broadcastAdd(Matrix* a, Matrix* b, Matrix* out);
    void broadcastMultiply(Matrix* a, Matrix* b, Matrix* out);
    void matMultiply(Matrix* a, Matrix* b, Matrix* out);
    void inPlaceScaleMatrix(Matrix* matrix, double n);
    void applyFunction(Matrix* matrix, double (*function)(double val));
    Matrix* getColumn(Matrix* m, int index);
    Matrix* getRow(Matrix* m, int index);
    double getValue(Matrix* m, int i, int j);
    void setValue(Matrix* m, int i, int j, double value);

#endif  //NEURALNETS_MATRIXGPU_CUH
#ifdef __cplusplus
}
#endif