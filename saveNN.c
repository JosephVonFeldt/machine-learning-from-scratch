//
// Created by joeyv on 6/12/2023.
//

//
//struct NeuralNetwork{
//    int inputCount;
//    int numHiddenLayers;
//    int outputCount;
//    NNLayer *inputLayer;
//    NNLayer *outputLayer;
//    double (*activation)(double);
//    double (*actToDerivative)(double);
//};

#include "saveNN.h"
void saveLayer(NNLayer *layer, int prevNodeCount, FILE *fptr);
void loadLayers(NeuralNetwork *nn, FILE *fptr);

void saveNetwork(NeuralNetwork *nn, char *fileName){
    FILE *fptr;
    fptr = fopen(fileName, "w");

    int prevNodeCount = nn->inputCount;
    NNLayer* currLayer = nn->inputLayer->nextLayer;
    NNLayer* nextLayer = currLayer->nextLayer;

    fprintf(fptr, "IN%i|LAYERS%i|OUT%i|", nn->inputCount, nn->numHiddenLayers, nn->outputCount);
    while (nextLayer != NULL) {
        saveLayer(currLayer, prevNodeCount, fptr);
        prevNodeCount = currLayer->nodeCount;
        currLayer = nextLayer;
        nextLayer = currLayer->nextLayer;
    }
    saveLayer(currLayer, prevNodeCount, fptr);
    // should probably rework activation function to save that somehow
    fclose(fptr);
}

void loadNetwork(NeuralNetwork *nn, char *fileName) {
    FILE *fptr;
    fptr = fopen(fileName, "r");

    int inputCount, numLayers, outputCount;
    char str1[10];
    char str2[10];
    char str3[10];
    char str4[10];
    char str5[10];
    char str6[10];
    double d;
    int rows;
    int cols;

    fscanf(fptr, "IN%i|LAYERS%i|OUT%i|", &inputCount, &numLayers, &outputCount);
    printf("Input Count:\t%i\nNumber of Layers:\t%i\nOutput count:\t%i\n", inputCount, numLayers, outputCount);
    nn->inputCount = inputCount;
    nn->numHiddenLayers = numLayers;
    nn->outputCount = outputCount;
    nn->activation = sigmoid;
    nn->actToDerivative = sigmoidToDerivative;
    nn->inputLayer = initInputLayer(inputCount);
    loadLayers(nn, fptr);

    fclose(fptr);

//    int prevNodeCount = nn->inputCount;
//    NNLayer* currLayer = nn->inputLayer->nextLayer;
//    NNLayer* nextLayer = currLayer->nextLayer;
//
//    fprintf(fptr, "IN%i|LAYERS%i|OUT%i|", nn->inputCount, nn->numHiddenLayers, nn->outputCount);
//    while (nextLayer != NULL) {
//        saveLayer(currLayer, prevNodeCount, fptr);
//        currLayer = nextLayer;
//        nextLayer = currLayer->nextLayer;
//    }
//    // should probably rework activation function to save that somehow
//    fclose(fptr);
}
//struct NNLayer{
//    int nodeCount;
//    Matrix* biases;
//    Matrix* activations;
//    Matrix* weights;
//    Matrix* biasesGrad;
//    Matrix* weightsGrad;
//    NNLayer *prevLayer;
//    NNLayer *nextLayer;
//};
void saveLayer(NNLayer *layer, int preNodeCount, FILE *fptr) {
    // Weights
    fprintf(fptr, "LAYER[ %i %i]", preNodeCount, layer->nodeCount);
    fprintf(fptr, "W[ ");
    for(int row = 0; row < preNodeCount; row++) {
        for(int col =0; col < layer->nodeCount; col++) {
            fprintf(fptr, "%f ", layer->weights->values[row ][ col]);
        }
    }
    fprintf(fptr, "]B[ ");
    // Biases
    for(int col =0; col < layer->nodeCount; col++) {
        fprintf(fptr, "%f ", layer->biases->values[0][col]);
    }
    fprintf(fptr, "]");
}

void loadLayers(NeuralNetwork *nn, FILE *fptr) {
    int rows;
    int cols;
    char w[1000];
    NNLayer *currLayer = nn->inputLayer;
    fscanf_s(fptr, "LAYER[ %i %i]W[", &rows, &cols);
    for (int i = 0; i <= nn->numHiddenLayers; i++) {
        //fgetpos((fptr);)
        double val;
        printf("LAYER[ %i| %i]\n", rows, cols);
        currLayer = addLayer(cols, currLayer);
        for(int rowIndex = 0; rowIndex < rows; rowIndex++) {
            for(int colIndex = 0; colIndex < cols; colIndex++){
                fscanf(fptr, " %lf ", &val);
                currLayer->weights->values[rowIndex][colIndex] = val;
                //setValue(currLayer->weights, rowIndex, colIndex, val);
            }
        }
        fscanf_s(fptr, " %s", w);
        for(int colIndex = 0; colIndex < cols; colIndex++){
            fscanf(fptr, " %lf", &val);
            //printf("%f\n", val);
            currLayer->biases->values[0][colIndex] = val;
        }
        fscanf_s(fptr, "%s", w);
        fscanf_s(fptr, "%i %i", &rows, &cols);
        fscanf_s(fptr, "%s", w);
    }
    nn->outputLayer = currLayer;
}
