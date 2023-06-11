cmake_minimum_required(VERSION 3.22)
project(NeuralNets CUDA C )

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 17)

#
# add_executable(NeuralNets NeuralNetwork.c NeuralNetwork.h main.c mnist.c mnist.h Matrix.c Matrix.h)
add_executable(NeuralNets NeuralNetwork.cu NeuralNetwork.cuh main.c mnist.cu mnist.cuh MatrixGPU.cu MatrixGPU.cuh)

set_target_properties(NeuralNets PROPERTIES LINKER_LANGUAGE CUDA)

