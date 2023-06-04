//
// Created by joeyv on 6/2/2023.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Matrix.h"

typedef struct Model Model;
float randFloat();
float sigmoid(float x);
float feedForward(Model *m, float x0, float x1);
float cost(Model *m);
void randM(Model *m, float a, float b);
void grad(Model *m, Model *g);
void train(Model *m, float rate);
void printCurrent(Model *m);

struct Model{
    float w00;
    float w01;
    float w10;
    float w11;

    float b0;
    float b1;

    float fw0;
    float fw1;

    float fb;
};

int trainingData[][3] = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0}

};


size_t training_size = sizeof (trainingData)/sizeof(trainingData[0]);

int main() {
    srand(time(0));
    //srand(69);
    float rate  = 1;
    Model m;
    Model g;
    randM(&m, 1, 0);
    float c = cost(&m);
    for (size_t i = 0; i < 10000000; i++){
        c = cost(&m);
        if(i%500000 == 0) {
            printCurrent(&m);
            printf("cost = %f\n", c);
        }
        train(&m, rate);
    }
    printCurrent(&m);
    printf("cost = %f\n", c);

    return 0;
}

int main1() {
    srand(69);
//
//    Matrix *m = initMatrix(3, 5);
//    printMatrix(m);
//    printf("\n");
//    randomizeMatrix(m, 2, .5);
//    printMatrix(m);
//    deleteMatrix(m);


    Matrix *a = initMatrix(2, 3);
    a->values[0][0] = 1;
    a->values[0][1] = 2;
    a->values[0][2] = 3;

    a->values[1][0] = 4;
    a->values[1][1] = 5;
    a->values[1][2] = 6;
    printMatrix(a);
    printf("\n\n");
    Matrix *b = initMatrix(3, 2);
    b->values[0][0] = 10;
    b->values[0][1] = 11;

    b->values[1][0] = 20;
    b->values[1][1] = 21;

    b->values[2][0] = 30;
    b->values[2][1] = 31;
    printMatrix(b);
    printf("\n\n");
    Matrix *out = initMatrix(2, 2);
    matMultiply(a, b, out);
    printMatrix(out);
    inPlaceScaleMatrix(out, .01);
    printMatrix(out);

    return 0;
}

float randFloat() {
    return (float)rand()/RAND_MAX;
}

float feedForward(Model *m, float x0, float x1){
    float a0 = sigmoid(m->w00 * x0 + m->w10 * x1 + m->b0);
    float a1 = sigmoid(m->w01 * x0 + m->w11 * x1 + m->b1);
    float fa = sigmoid(m->fw0 * a0 + m->fw1 * a1 + m->fb);
    return fa;

}
float cost(Model *m) {
    float total = 0;
    float ff;
    for(size_t i=  0; i < training_size; i++) {
        ff = feedForward(m, trainingData[i][0], trainingData[i][1]) - trainingData[i][2];
        total += ff*ff;
    }
    return total;
}

void grad(Model *m, Model *g) {
    float x0, x1, y, a0, a1, fa = 0;
    float dw00=0, dw10=0, dw01=0, dw11=0, dfw0=0, dfw1 = 0;
    float db0=0, db1=0, dbf = 0;
    for(int i=  0; i < training_size; i++){
        x0 = trainingData[i][0];
        x1 = trainingData[i][1];
        y = trainingData[i][2];

        a0 = sigmoid(m->w00 * x0 + m->w10 * x1 + m->b0);
        a1 = sigmoid(m->w01 * x0 + m->w11 * x1 + m->b1);
        fa = sigmoid(m->fw0 * a0 + m->fw1 * a1 + m->fb);

        float temp_dbf = 2 * (fa-y) * fa * (1-fa );
        dbf += temp_dbf;

        dfw1 += temp_dbf * a1;
        dfw0 += temp_dbf * a0;

        float temp_db1 = 2 * (fa-y) * fa * (1-fa) * m->fw1 * a1 * (1-a1);
        db1 += temp_db1;
        dw11 += temp_db1 * x1;
        dw01 += temp_db1 * x0;

        float temp_db0 = 2 * (fa-y) * fa * (1-fa) * m->fw0 * a0 * (1-a0);
        db0 += temp_db0;
        dw10 += temp_db0 * x1;
        dw00 += temp_db0 * x0;


    }
    dbf /= training_size;
    dfw1 /= training_size;
    dfw0 /= training_size;

    db1 /= training_size;
    dw11 /= training_size;
    dw10 /= training_size;

    db0 /= training_size;
    dw01 /= training_size;
    dw00 /= training_size;

    g->w00 = dw00;
    g->w01 = dw01;

    g->w10 = dw10;
    g->w11 = dw11;

    g->fw0 = dfw0;
    g->fw1 = dfw1;

    g->b0 = db0;
    g->b1 = db1;

    g->fb = dbf;
}

void train(Model *m, float rate) {
    Model g;
    g.w00 = 0;
    g.w01 = 0;

    g.w10 = 0;
    g.w11 = 0;

    g.fw0 = 0;
    g.fw1 = 0;

    g.b0 = 0;
    g.b1 = 0;

    g.fb = 0;

    grad(m, &g);

    m->w00 -= g.w00 * rate;
    m->w01 -= g.w01 * rate;

    m->w10 -= g.w10 * rate;
    m->w11 -= g.w11 * rate;

    m->fw0 -= g.fw0 * rate;
    m->fw1 -= g.fw1 * rate;

    m->b0 -= g.b0 * rate;
    m->b1 -= g.b1 * rate;

    m->fb -= g.fb * rate;
}

void randM(Model *m, float a, float b) {
    m->w00 = a * randFloat() - b;
    m->w01 =  a * randFloat()- b;

    m->w10 =  a * randFloat()- b ;
    m->w11 = a * randFloat()- b;
    m->fw0 = a * randFloat()- b;
    m->fw1 = a * randFloat()- b ;

    m->b0 = a * randFloat()- b;
    m->b1 = a * randFloat()- b ;

    m->fb = a * randFloat()-b ;
}

float sigmoid(float x) {
    return 1/(1+expf(-x));
}

void printCurrent(Model *m) {
    int x0, x1, y;
    float c, out;
    for(int i=  0; i < training_size; i++){
        x0 = trainingData[i][0];
        x1 = trainingData[i][1];
        y = trainingData[i][2];
        out = feedForward(m, x0, x1);
        printf("In1: %i\tIn2: %i\t out: %f\t expected: %i\n",x0, x1, out, y);
    }
    printf("\n\n");
}