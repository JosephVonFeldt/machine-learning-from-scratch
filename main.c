//
// Created by joeyv on 6/3/2023.
//
#include "NeuralNetwork.h"
#include "mnist.h"
#include <stdlib.h>
#include <stdio.h>

void currentStateMNIST(NeuralNetwork *nn, Matrix* trainingInputs, Matrix* trainingExpected);
void currentState(NeuralNetwork *nn, Matrix* trainingInputs, Matrix* trainingExpected);
int main() {
    int example = 2;
    if (example == 1){
        NeuralNetwork* nn = initNetwork(2, 2, 1, 3);

        Matrix* trainingInput = initMatrix(4,2);
        setValue(trainingInput, 0, 0, 0);
        setValue(trainingInput, 0, 1, 0);

        setValue(trainingInput, 1, 0, 0);
        setValue(trainingInput, 1, 1, 1);

        setValue(trainingInput, 2, 0, 1);
        setValue(trainingInput, 2, 1, 0);

        setValue(trainingInput, 3, 0, 1);
        setValue(trainingInput, 3, 1, 1);
        Matrix* trainingAnswers = initMatrix(1, 4);

        setValue(trainingAnswers, 0, 0, 0);
        setValue(trainingAnswers, 0, 1, 1);
        setValue(trainingAnswers, 0, 2, 1);
        setValue(trainingAnswers, 0, 3, 0);

        for (int i = 0; i < 1e6+1; i++) {
            train(nn, trainingInput, trainingAnswers, .1, 1);
            if (i%50000 == 0) {
                currentState(nn,trainingInput, trainingAnswers);
                printf("************************************************************************************\n");
            }
        }
    }
    else{
        NeuralNetwork* nn = initNetwork(784, 2, 10, 30);
        // You may need to mess with the file path
        char* filename = ".\\..\\..\\machine-learning-from-scratch\\MNIST\\mnist_train.csv";//
        int lines = lineCount(filename);
        Matrix* trainingInput = initMatrix(lines,784);
        Matrix* trainingAnswers = initMatrix(10, lines);
        getMnistFileData(trainingInput, trainingAnswers,filename);

        filename = ".\\..\\..\\machine-learning-from-scratch\\MNIST\\mnist_test.csv";//
        lines = lineCount(filename);
        Matrix* testInput = initMatrix(lines,784);
        Matrix* testAnswers = initMatrix(10, lines);
        getMnistFileData(testInput, testAnswers,filename);

        for (int i = 0; i < 1e6+1; i++) {
            printf("%i\n", i);
            train(nn, trainingInput, trainingAnswers, .1, 0 );
            if (i%10 == 0 ) {
                currentStateMNIST(nn, testInput, testAnswers);
                printf("**********************************\n");
                printf("%i\n", i);
                printf("**********************************\n");
            }
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
        printf("IN: %f\t%f\t|\tOUT: %f\t\tExpected: %f\n\n",getValue(row, 0, 0), getValue(row, 0, 1),
               getValue(actual, 0, 0), getValue(expected, 0, 0));
        deleteMatrix(row);
        deleteMatrix(actual);
        deleteMatrix(expected);
    }
}
void printInputLayer(Matrix* row) {
    printf("**********************************\n\n");
    for (int i =0; i < row->columns; i++) {

        if(getValue(row, 0, i) < .1){
            printf(" ");
        }else {
            if(getValue(row, 0, i) < .5){
                printf(".");
            }else {
                printf("*");
            }
        }
        if(i%28 == 0){
            printf("\n");
        }

    }
    printf("\n");
}

void currentStateMNIST(NeuralNetwork *nn, Matrix* trainingInputs, Matrix* trainingExpected) {
    for (int i =0; i < trainingInputs->rows; i++) {
        if(((double)rand())/RAND_MAX < .001){
            Matrix* row = getRow(trainingInputs,i);
            printInputLayer(row);
            setInput(nn, row);

            feedForward(nn);
            Matrix* expected = getColumn(trainingExpected, i);
            Matrix* actual = getOutput(nn);
            printf("  Actual \t|Predicted\n");
            printf("________________|________________\n");
            for(int j=0; j<10;j++) {
                printf("%i: %f\t|%f\n", j, getValue(expected, j, 0), getValue(actual, 0, j));
            }
            printf("\n");
            deleteMatrix(row);
            deleteMatrix(actual);
            deleteMatrix(expected);
        }

    }
}
