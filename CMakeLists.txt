cmake_minimum_required(VERSION 3.25)
project(NeuralNets C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralNets Matrix.c Matrix.h NeuralNetwork.c NeuralNetwork.h main.c mnist.c mnist.h)
