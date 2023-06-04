//
// Created by joeyv on 6/3/2023.
//

#include <math.h>
#include "NeuralNetwork.h"

void descend(NeuralNetwork* nn);
//struct NeuralNetwork{
//    int inputCount;
//    int numLayers;
//    int outputCount;
//    NNLayer *inputLayer;
//    NNLayer *outputLayer;
//};
//NeuralNetwork* initNetwork(int inputCount, int numLayers, int outputCount, int nodesPerLayer);
//
//struct NNLayer{
//    int nodeCount;
//    Matrix* biases;
//    Matrix* activations;
//    Matrix* weightsGrad;
//    Matrix* weights;
//    NNLayer *prevLayer;
//    NNLayer *nextLayer;
//};

double randMax = 1.0;
double randMin = -0.3;
NeuralNetwork* initNetwork(int inputCount, int numHiddenLayers, int outputCount, int nodesPerLayer) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->inputCount = inputCount;
    nn->numHiddenLayers = numHiddenLayers;
    nn->outputCount = outputCount;
    nn->inputLayer = initInputLayer(inputCount);
    nn->activation = sigmoid;
    nn->actToDerivative = sigmoidToDerivative;
    NNLayer* currLayer = nn->inputLayer;

    for (int i = 0; i < numHiddenLayers; i++) {
        currLayer = addLayer(nodesPerLayer, currLayer);
    }
    nn->outputLayer = addLayer(outputCount, currLayer);
    return nn;
}

NNLayer* initInputLayer(int inputCount) {
    NNLayer* layer = malloc(sizeof(NNLayer));
    layer->nodeCount = inputCount;
    layer->prevLayer = NULL;
    layer->weightsGrad = NULL;
    layer->biasesGrad = NULL;
    layer->weights = NULL;
    layer->biases = NULL;
    layer->nextLayer = NULL;

    layer->activations = initMatrix(1, inputCount);
    return layer;
}

NNLayer* addLayer(int nodeCount, NNLayer *prevLayer) {
    // init new layer
    NNLayer* layer = malloc(sizeof(NNLayer));
    layer->nodeCount = nodeCount;
    layer->prevLayer = prevLayer;
    layer->weightsGrad = initMatrix(prevLayer->nodeCount, nodeCount);
    layer->weights = initMatrix(prevLayer->nodeCount, nodeCount);
    layer->biasesGrad = initMatrix(1, nodeCount);
    layer->biases = initMatrix(1, nodeCount);
    layer->activations = initMatrix(1, nodeCount);
    layer->nextLayer = NULL;

    // randomize weights and biases
    randomizeMatrix(layer->weights, randMax,randMin);
    randomizeMatrix(layer->biases, randMax,randMin);

    // update prevLayer's 'nextLayer' pointer
    prevLayer->nextLayer = layer;

    return layer;
}

NNLayer* deleteLayer(NNLayer *layer) {
    NNLayer* nextLayer = layer->nextLayer;
    deleteMatrix(layer->weightsGrad);
    deleteMatrix(layer->biasesGrad);
    deleteMatrix(layer->weights);
    deleteMatrix(layer->biases);
    deleteMatrix(layer->activations);
    free(layer);
    return nextLayer;
}

void feedForward(NeuralNetwork *nn) {
    NNLayer* currLayer = nn->inputLayer;
    NNLayer* nextLayer = currLayer->nextLayer;
    while (nextLayer != NULL) {
        // Weights and biases
        matMultiply(currLayer->activations, nextLayer->weights, nextLayer->activations);
        broadcastAdd(nextLayer->activations, nextLayer->biases, nextLayer->activations);
        // Applying activation function
        applyFunction(nextLayer->activations, nn->activation);
        currLayer = nextLayer;
        nextLayer = currLayer->nextLayer;
    }
}

Matrix* getOutput(NeuralNetwork *nn) {
    Matrix* output = initMatrix(1, nn->outputCount);
    copyInto(nn->outputLayer->activations, output);
    return output;
}

void train (NeuralNetwork *nn, Matrix* trainingInputs, Matrix* trainingExpected, double rate) { //Data
    assert(trainingInputs->rows == trainingExpected->columns);
    Matrix* biasDerHolder;
    Matrix* nextBiasHolder;
    double r = rate / trainingInputs->rows;
    for (int i =0; i < trainingInputs->rows; i++) {
        Matrix* row = getRow(trainingInputs,i);
        setInput(nn, row);
        deleteMatrix(row);
        feedForward(nn);
        Matrix* expected = getColumn(trainingExpected, i);
        Matrix* actual = getOutput(nn);
        NNLayer* currLayer = nn->outputLayer;
        biasDerHolder = initMatrix(1, currLayer->nodeCount);
        while (currLayer->prevLayer != NULL) {
            nextBiasHolder = initMatrix(1, currLayer->prevLayer->nodeCount);
            double oneDivNodeCount = 1./currLayer->nodeCount;
            for (int j = 0; j < currLayer->nodeCount; j++) {
                if (currLayer == nn->outputLayer){
                    biasDerHolder->values[0][j] = r * (actual->values[0][j] - expected->values[0][j]) * nn->actToDerivative(actual->values[0][j]);
                }
                else {
                    biasDerHolder->values[0][j] *= nn->actToDerivative(currLayer->activations->values[0][j]) ;
                }
                currLayer->biasesGrad->values[0][j] += biasDerHolder->values[0][j];
                for(int k = 0; k < currLayer->weightsGrad->rows; k++) {
                    currLayer->weightsGrad->values[k][j] += oneDivNodeCount * biasDerHolder->values[0][j] * currLayer->prevLayer->activations->values[0][k];
                    nextBiasHolder->values[0][k] += oneDivNodeCount * biasDerHolder->values[0][j] * currLayer->weights->values[k][j];
                }
            }
            deleteMatrix(biasDerHolder);
            biasDerHolder = nextBiasHolder;
            currLayer = currLayer->prevLayer;
        }
        deleteMatrix(actual);
        deleteMatrix(expected);
        deleteMatrix(nextBiasHolder);
    }
    descend(nn);
}

void descend(NeuralNetwork* nn) {
    NNLayer* currLayer = nn->inputLayer->nextLayer;
    while (currLayer != NULL) {
        for (int colIndex = 0; colIndex < currLayer->weights->columns; colIndex++) {
            for (int rowIndex = 0; rowIndex < currLayer->weights->rows; rowIndex++) {
                currLayer->weights->values[rowIndex][colIndex] -= currLayer->weightsGrad->values[rowIndex][colIndex];
                currLayer->weightsGrad->values[rowIndex][colIndex] = 0;
            }
            currLayer->biases->values[0][colIndex] -= currLayer->biasesGrad->values[0][colIndex];
            currLayer->biasesGrad->values[0][colIndex] = 0;
        }
        currLayer = currLayer->nextLayer;
    }
}

void setInput(NeuralNetwork* nn, Matrix* input) {
    assert(input->rows == 1);
    assert(input->columns == nn->inputCount);
    copyInto(input, nn->inputLayer->activations);
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
double sigmoidToDerivative(double x) {
    return x * (1-x);
}