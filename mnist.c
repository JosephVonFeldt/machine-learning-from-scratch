//
// Created by joeyv on 6/4/2023.
//

#include <stdio.h>
#include "mnist.h"


int main1() {
    FILE *fptr;
    char* fileName = ".\\..\\..\\machine-learning-from-scratch\\MNIST\\mnist_train.csv";
    // Open a file in read mode
    printf("%i\n", lineCount(fileName));
   // return 0;
    char myString[4000];
    int i = 0;
    fptr = fopen(fileName, "r");
    if (fptr == NULL) {
        return 1;
    }
    int count = 0;
    int num = 0;
    // Read the content and print it
    while(fgets(myString, 4000, fptr) && i< 10000) {
        if (i <10000){
            for(int j=0; j< 4000; j++){
                if (myString[j] >= 48  && myString[j] < 58)
                num = num * 10 + myString[j] - 48;
                if (myString[j]==','){
                    count++;
                    if (count == 784) {
                        break;
                    }
                    num = 0;
                }
            }
            printf("%i\t%i\n",num,  count++);
            count = 0;

        }
        i++;
    }

    // Close the file
    fclose(fptr);
    return 0;
}

int lineCount(char* fileName) {
    FILE *fptr;
    // Open a file in read mode
    fptr = fopen(fileName, "r");
    // Store the content of the file
    char myString[4000];
    int i = 0;
    if (fptr == NULL) {
        return 0;
    }
    int count = 0;
    // Read the content and print it
    while(fgets(myString, 4000, fptr)) {
        count++;
    }

    // Close the file
    fclose(fptr);
    return count;
}
void getMnistFileData(Matrix* input, Matrix* output, char* filename) { // this assumes that expected output is followed by 784 input values
    assert(output->rows == 10);
    assert(input->columns == 784);
    assert(output->columns == input->rows);

    FILE *fptr;
    // Open a file in read mode
    fptr = fopen(filename, "r");
    assert(fptr != NULL);
    // Store the content of the file
    char myString[4000];
    int numLines = output->columns;
    int count = 0;
    int num = 0;

    for(int currLine=0; currLine<numLines; currLine++){
        fgets(myString, 4000, fptr);
        for(int j=0; j< 4000; j++){
            if (myString[j] >= 48  && myString[j] < 58)
                num = num * 10 + myString[j] - 48;
            if (myString[j]==','){
                if(count == 0){
                    output->values[num][currLine] = 1;
                } else {
                    input->values[currLine][count-1] = ((double)num)/255;
                }
                count++;
                if (count == 784) {
                    break;
                }
                num = 0;
            }
        }
        count = 0;
    }
    fclose(fptr);


}