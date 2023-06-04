cmake_minimum_required(VERSION 3.25)
project(HelloWorld C)

set(CMAKE_C_STANDARD 11)

add_executable(HelloWorld Matrix.c Matrix.h NeuralNetwork.c NeuralNetwork.h main.c)
