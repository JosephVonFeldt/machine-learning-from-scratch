//
// Created by joeyv on 6/3/2023.
//

#ifndef NeuralNets_NEURALNETWORK_H
#define NeuralNets_NEURALNETWORK_H
#include "MatrixGPU.cuh"
//#include "Matrix.h"
typedef struct NeuralNetwork NeuralNetwork;
typedef struct NNLayer NNLayer;
double sigmoid(double x);
double sigmoidToDerivative(double
x);
struct NeuralNetwork{
    int inputCount;
    int numHiddenLayers;
    int outputCount;
    NNLayer *inputLayer;
    NNLayer *outputLayer;
    double (*activation)(double);
    double (*actToDerivative)(double);
};
Matrix* getOutput(NeuralNetwork *nn);
void feedForward(NeuralNetwork *nn);
void train (NeuralNetwork *nn, Matrix* trainingInputs, Matrix* trainingExpected, double rate, int all);
void setInput(NeuralNetwork* nn, Matrix* input);
NeuralNetwork* initNetwork(int inputCount, int numLayers, int outputCount, int nodesPerLayer);
NNLayer* initInputLayer(int nodes);
NNLayer* addLayer(int nodeCount, NNLayer *prevLayer);
NNLayer* deleteLayer(NNLayer *layer);
struct NNLayer{
    int nodeCount;
    Matrix* biases;
    Matrix* activations;
    Matrix* weights;
    Matrix* biasesGrad;
    Matrix* weightsGrad;
    NNLayer *prevLayer;
    NNLayer *nextLayer;
};
#endif //NeuralNets_NEURALNETWORK_H
