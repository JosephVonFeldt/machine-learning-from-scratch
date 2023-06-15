//
// Created by joeyv on 6/12/2023.
//

#ifndef NEURALNETS_SAVENN_H
#define NEURALNETS_SAVENN_H
#include "NeuralNetwork.h"
#include <stdio.h>
void saveNetwork(NeuralNetwork *nn, char *fileName);
void loadNetwork(NeuralNetwork *nn, char *fileName);
#endif //NEURALNETS_SAVENN_H
