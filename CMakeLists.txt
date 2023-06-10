cmake_minimum_required(VERSION 3.22)
project(NeuralNets CUDA C )

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 17)

#Matrix.c Matrix.h
add_executable(NeuralNets NeuralNetwork.c NeuralNetwork.h main.c mnist.c mnist.h MatrixGPU.cu MatrixGPU.cuh)

set_target_properties(NeuralNets PROPERTIES LINKER_LANGUAGE CUDA)

