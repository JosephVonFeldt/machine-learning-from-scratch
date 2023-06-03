#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
typedef struct Model Model;
float randFloat();
float feedForward(Model *m, float x);
float cost(Model *m);
void randM(Model *m);
void grad (Model *m, Model *g);
void train(Model *m, float rate);

struct Model{
    float w;
    float b;
};

int trainingData[][2] = {
    {91, 456},
    {45, 226},
    {51, 256},
    {59, 296},
    {100, 501},
    {53, 266},
    {67, 336},
    {86, 431},
    {47, 236},
    {1, 6},
    {37, 186},
    {35, 176},
    {15, 76},
    {95, 476},
    {76, 381},
    {7, 36},
    {85, 426},
    {94, 471},
    {4, 21},
    {61, 306}
};
size_t training_size = sizeof (trainingData)/sizeof(trainingData[0]);

int main() {
    // srand(time(0));
    srand(69);
    float rate  = .0002;
    Model m;
    Model g;
    randM(&m);
    float c = cost(&m);
    for (size_t i = 0; i < 100000000; i++){
        g.w = 0;
        g.b = 0;
        c = cost(&m);


        train(&m, rate);
    }
    printf("w = %f\tb = %f\tcost = %f", m.w, m.b, c);

    return 0;
}

float randFloat() {
    return (float)rand()/RAND_MAX;
}

float feedForward(Model *m, float x){
    return m->w * x + m->b;
}
float cost(Model *m) {
    float total = 0;
    float ff = 0;
    for(size_t i=  0; i < training_size; i++){
        ff = feedForward(m, trainingData[i][0]) - trainingData[i][1];
        total += ff*ff;
    }
    return total;
}

void grad(Model *m, Model *g) {
    float total_w = 0;
    float total_b = 0;
    for(size_t i=  0; i < training_size; i++){
        total_w += 2 * (feedForward(m, trainingData[i][0]) - trainingData[i][1]) * trainingData[i][0];
        total_b += 2 * (feedForward(m, trainingData[i][0]) - trainingData[i][1]);
    }
    total_w /= training_size;
    total_b /= training_size;
    g->w = total_w;
    g->b = total_b;
}

void train(Model *m, float rate) {
    Model g;
    g.w = 0;
    g.b = 0;
    grad(m, &g);
    m->w-= g.w * rate;
    m->b-= g.b * rate * 3;
}
void randM(Model *m) {
    m->w = 10 * randFloat() - 5;
    m->b = 10 * randFloat() - 5;
}
