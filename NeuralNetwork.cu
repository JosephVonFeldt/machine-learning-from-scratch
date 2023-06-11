//
// Created by joeyv on 6/3/2023.
//

#include <math.h>
#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>

#include "NeuralNetwork.cuh"

void descend(NeuralNetwork* nn);
__device__ double sigmoidToDerivativeGPU(double x);

double randMax = 0.50;
double randMin = -0.50;
NeuralNetwork* initNetwork(int inputCount, int numHiddenLayers, int outputCount, int nodesPerLayer) {
    NeuralNetwork *nn = static_cast<NeuralNetwork *>(malloc(sizeof(NeuralNetwork)));
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
    NNLayer* layer = static_cast<NNLayer *>(malloc(sizeof(NNLayer)));
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
    NNLayer* layer = static_cast<NNLayer *>(malloc(sizeof(NNLayer)));
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
        //sync();
        applyFunction(nextLayer->activations, nn->activation);
        currLayer = nextLayer;
        nextLayer = currLayer->nextLayer;
    }
    //sync();
}

Matrix* getOutput(NeuralNetwork *nn) {
    Matrix* output = initMatrix(1, nn->outputCount);
    copyInto(nn->outputLayer->activations, output);
    return output;
}

__global__ void calcFirstBiasPartials(double *biasDerHolder, double *actual, double *expected, double *cumulativeBiasPartials, double r, int size) {
//    biasPartialDerivative = r * (getValue(actual, 0, j) - getValue(expected, j, 0)) * nn->actToDerivative(getValue(actual, 0, j));
//    setValue(biasDerHolder, 0, j, biasPartialDerivative);
//    if(getValue(expected, j, 0) > 0){
//        biasPartialDerivative *= expected->rows -1 ;//maybe rows -1 would be better?
//        setValue(biasDerHolder, 0, j, biasPartialDerivative);
//    }
    unsigned int num = blockDim.x * blockIdx.x + threadIdx.x;
    if (num < size) {
        double x = r * (actual[num] - expected[num]) * sigmoidToDerivativeGPU(actual[num]);
        if (expected[num] > 0) {
            x *= size;
        }
        biasDerHolder[num] = x;
        cumulativeBiasPartials[num] += x;
    }
}


__global__ void calcBiasPartials(double *biasDerHolder, double *currActivations, double *cumulativeBiasPartials, int size) {
//    biasPartialDerivative = r * (getValue(actual, 0, j) - getValue(expected, j, 0)) * nn->actToDerivative(getValue(actual, 0, j));
//    setValue(biasDerHolder, 0, j, biasPartialDerivative);
//    if(getValue(expected, j, 0) > 0){
//        biasPartialDerivative *= expected->rows -1 ;//maybe rows -1 would be better?
//        setValue(biasDerHolder, 0, j, biasPartialDerivative);
//    }
    unsigned int num = blockDim.x * blockIdx.x + threadIdx.x;
    if (num < size) {
        biasDerHolder[num] *= currActivations[num];
        cumulativeBiasPartials[num] += biasDerHolder[num];
    }
}

__global__ void calcWeightsPartials(double *biasDerHolder, double *prevActivations, double *nextBiasPartials, double *currWeightsPartials, double *currWeights, int rows, int cols) {
    //   for(int k = 0; k < currLayer->weightsGrad->rows; k++) {
//                        double currWeightsPartial = biasPartialDerivative * getValue(currLayer->prevLayer->activations, 0, k);
//                        setValue(currLayer->weightsGrad, k, j, currWeightsPartial); //+=
//                        double nextBiasPartialPiece = biasPartialDerivative * getValue(currLayer->weights, k, j);
//                        setValue(nextBiasHolder,0, k, nextBiasPartialPiece); //+=
    unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < cols) {
        for(int k = 0; k < rows; k++) {
            double currWeightsPartial = biasDerHolder[j] * prevActivations[k];
            currWeightsPartials[k * cols + j ] = currWeightsPartial;
            double nextBiasPartialPiece = biasDerHolder[j] * currWeights[k * cols + j];
            nextBiasPartials[k] = nextBiasPartialPiece;
        }
    }
}

__global__ void trainGPU(NeuralNetwork *nn, Matrix *row, Matrix *expected, Matrix *c ) {
//    setInput(nn, row);
//    deleteMatrix(row);
//    feedForward(nn);
//    Matrix* expected = getColumn(trainingExpected, i);
//    Matrix* actual = getOutput(nn);
//    NNLayer* currLayer = nn->outputLayer;
//    biasDerHolder = initMatrix(1, currLayer->nodeCount);
//    while (currLayer->prevLayer != NULL) {
//        nextBiasHolder = initMatrix(1, currLayer->prevLayer->nodeCount);
//        double oneDivNodeCount = 1./currLayer->nodeCount;
//        for (int j = 0; j < currLayer->nodeCount; j++) {
//            // Region
//            // __global__ void calcFirstBiasPartials(double *biasDerHolder, double *actual, double *cumulativeBiasPartials, double *expected, double r, int size)
//            double biasPartialDerivative;
//            if (currLayer == nn->outputLayer){
//                calcFirstBiasPartials<<<currLayer->nodeCount, 1>>>(biasDerHolder->values, actual->values, expected->values, currLayer->biasesGrad->values, r, currLayer->nodeCount);
//            }
//            else {
//                calcBiasPartials<<<currLayer->nodeCount, 1>>>(biasDerHolder->values, currLayer->activations->values, currLayer->biasesGrad->values, currLayer->nodeCount);
//            }
//            cudaDeviceSynchronize();
//            calcWeightsPartials<<<currLayer->nodeCount, 1>>>(biasDerHolder->values, currLayer->prevLayer->activations->values, nextBiasHolder->values, currLayer->weightsGrad->values, currLayer->weights->values, currLayer->weightsGrad->rows, currLayer->nodeCount);
//        }
//        deleteMatrix(biasDerHolder);
//        biasDerHolder = nextBiasHolder;
//        currLayer = currLayer->prevLayer;
//    }
//    deleteMatrix(actual);
//    deleteMatrix(expected);
//    deleteMatrix(nextBiasHolder);
//    cudaDeviceSynchronize();
//    descend(nn);
//    cudaDeviceSynchronize();
//
}

void train (NeuralNetwork *nn, Matrix* trainingInputs, Matrix* trainingExpected, double rate, int all) { //Data
    assert(trainingInputs->rows == trainingExpected->columns);
    Matrix* biasDerHolder;
    Matrix* nextBiasHolder;
    double r = rate;// / nn->numHiddenLayers;
    for (int i =0; i < trainingInputs->rows; i++) {
        if (((double)rand())/RAND_MAX < .003 || all){
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
                    // Region
                    // __global__ void calcFirstBiasPartials(double *biasDerHolder, double *actual, double *cumulativeBiasPartials, double *expected, double r, int size)
                    double biasPartialDerivative;
                    if (currLayer == nn->outputLayer){
                        calcFirstBiasPartials<<<currLayer->nodeCount, 1>>>(biasDerHolder->values, actual->values, expected->values, currLayer->biasesGrad->values, r, currLayer->nodeCount);
                    }
                    else {
                        calcBiasPartials<<<currLayer->nodeCount, 1>>>(biasDerHolder->values, currLayer->activations->values, currLayer->biasesGrad->values, currLayer->nodeCount);
                    }
                    cudaDeviceSynchronize();
                    calcWeightsPartials<<<currLayer->nodeCount, 1>>>(biasDerHolder->values, currLayer->prevLayer->activations->values, nextBiasHolder->values, currLayer->weightsGrad->values, currLayer->weights->values, currLayer->weightsGrad->rows, currLayer->nodeCount);
                }
                deleteMatrix(biasDerHolder);
                biasDerHolder = nextBiasHolder;
                currLayer = currLayer->prevLayer;
            }
            deleteMatrix(actual);
            deleteMatrix(expected);
            deleteMatrix(nextBiasHolder);
            cudaDeviceSynchronize();
            descend(nn);
            cudaDeviceSynchronize();
        }
    }
}

__global__ void descend(double *weights, double *biases, double *weightsGrad, double *biasesGrad ,int rowCount, int colCount) {
    unsigned int num = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = num/colCount;
    unsigned int col = num%colCount;
    if (col < colCount && row < rowCount) {
        double newWeight = weights[row * colCount + col] - weightsGrad[row * colCount + col];
        weights[row * colCount + col] = newWeight;
        if (row == 0) {
            double newBias = biases[row * colCount + col] - biasesGrad[row * colCount + col];
            biases[row * colCount + col] = newBias;
        }

    }
}



void descend(NeuralNetwork* nn) {
    NNLayer* currLayer = nn->inputLayer->nextLayer;
    while (currLayer != NULL) {
        descend<<<currLayer->weights->rows, currLayer->weights->columns>>>(currLayer->weights->values, currLayer->biases->values, currLayer->weightsGrad->values, currLayer->biasesGrad->values , currLayer->weights->rows, currLayer->weights->columns);
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

__device__ double sigmoidToDerivativeGPU(double x) {
    return x * (1-x);
}