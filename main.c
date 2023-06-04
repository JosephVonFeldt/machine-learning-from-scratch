//
// Created by joeyv on 6/3/2023.
//
#include "NeuralNetwork.h"
void currentState(NeuralNetwork *nn, Matrix* trainingInputs, Matrix* trainingExpected);
int main() {

    NeuralNetwork* nn = initNetwork(2, 2, 1, 3);

    Matrix* trainingInput = initMatrix(4,2);
    trainingInput->values[0][0] = 0;
    trainingInput->values[0][1] = 0;

    trainingInput->values[1][0] = 0;
    trainingInput->values[1][1] = 1;

    trainingInput->values[2][0] = 1;
    trainingInput->values[2][1] = 0;

    trainingInput->values[3][0] = 1;
    trainingInput->values[3][1] = 1;
    Matrix* trainingAnswers = initMatrix(1, 4);

    trainingAnswers->values[0][0] = 0;
    trainingAnswers->values[0][1] = 1;
    trainingAnswers->values[0][2] = 1;
    trainingAnswers->values[0][3] = 0;

    for (int i = 0; i < 1e6+1; i++) {
        train(nn, trainingInput, trainingAnswers, 1);
        if (i%50000 == 0) {
            currentState(nn,trainingInput, trainingAnswers);
            printf("************************************************************************************\n");
        }
    }
    return 0;
}


void currentState(NeuralNetwork *nn, Matrix* trainingInputs, Matrix* trainingExpected) {
    for (int i =0; i < trainingInputs->rows; i++) {
        Matrix* row = getRow(trainingInputs,i);
        setInput(nn, row);

        feedForward(nn);
        Matrix* expected = getColumn(trainingExpected, i);
        Matrix* actual = getOutput(nn);
        printf("IN: %f\t%f\t|\tOUT: %f\t\tExpected: %f\n\n",row->values[0][0], row->values[0][1],
                actual->values[0][0], expected->values[0][0]);
        deleteMatrix(row);
        deleteMatrix(actual);
        deleteMatrix(expected);
    }
}