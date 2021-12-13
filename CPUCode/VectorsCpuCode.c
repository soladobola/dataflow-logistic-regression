/**
 * Document: MaxCompiler Tutorial (maxcompiler-tutorial.pdf)
 * Chapter: 6      Example: 3      Name: Vectors
 * MaxFile name: Vectors
 * Summary:
 *    Streams a vector of integers to the dataflow engine and confirms that the
 *    returned stream contains the integers values doubled.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include "Maxfiles.h"
#include <MaxSLiCInterface.h>

#define MAX_LINE_WIDTH 255


const int vectorSize = 8;

struct Dataset{
    float** features;
    float* labels;
    int feature_count;
    int examples;
};


int CountLines(const char* filename){
    FILE *fp;
    fp = fopen(filename, "r");
    int count = 0;
    char line[MAX_LINE_WIDTH];
    while(!feof(fp)){
        fgets(line, MAX_LINE_WIDTH, fp);
        count++;
    }
    fclose(fp);
    return count;
}

struct Dataset AllocateMemory(const char* filename){
    int data_count = CountLines(filename);
    int feature_count = 1; 
    FILE *fp;
    fp = fopen(filename, "r");
    char line[MAX_LINE_WIDTH];
    fgets(line, MAX_LINE_WIDTH, fp);
    for(int i = 0; i < MAX_LINE_WIDTH; i++){
        if(line[i] == '\n') break;
        if(isspace(line[i]))
            feature_count++;
    }
    
    struct Dataset dataset;
    dataset.feature_count = feature_count;
    dataset.examples = data_count;
    dataset.features = (float**)malloc(data_count*sizeof(float*));
    for(int i = 0; i < data_count; i++)
        dataset.features[i] = (float*)malloc(feature_count*sizeof(float));
    dataset.labels = (float*)malloc(data_count*sizeof(float));
    
    fclose(fp);
    return dataset;
}


void ReadDataset(const char* filename, struct Dataset* dataset){
    FILE *fp = fopen(filename, "r");
    int label_index = 0;
    
    
    for(int i = 0; i < dataset->examples; i++){
        for(int j = 0; j < dataset->feature_count; j++){
            
            if(j == dataset->feature_count - 1){
               fscanf(fp, "%f", &(dataset->labels[label_index++]));
               continue;
             
            }
            
            fscanf(fp, "%f", &(dataset->features[i][j]));
            
        }
    }
    dataset->feature_count--;
    
    fclose(fp);
}


struct Dataset FeatureMap(struct Dataset dataset, int degree){
    struct Dataset new_dataset;
    new_dataset.feature_count = dataset.feature_count*degree + 1;
    new_dataset.examples = dataset.examples;
    new_dataset.features = (float**)malloc(dataset.examples*sizeof(float*));
    for(int i = 0; i < dataset.examples; i++){
        new_dataset.features[i] = (float*)malloc(new_dataset.feature_count*sizeof(float));
    }
    new_dataset.labels = dataset.labels;
    
    for(int i = 0; i < dataset.examples; i++){
        new_dataset.features[i][0] = 1;
    }
    
    for(int item = 0; item < dataset.feature_count; item++){
        for(int k = 1; k <= degree; k++){
            for(int row = 0; row < dataset.examples; row++){
                new_dataset.features[row][1 + dataset.feature_count*(k-1) + item] = pow(dataset.features[row][item], k);
            }
        }
    }
    
    return new_dataset;
}

void PrintDataset(struct Dataset dataset){
    for(int i = 0; i < dataset.examples; i++){
        for(int j = 0; j < dataset.feature_count; j++){
            printf("%f ", dataset.features[i][j]);
        }
        printf("\n");
    }
}



float* GradientDescentCPU(struct Dataset dataset, float* theta){
    float *vect = (float*)malloc(dataset.feature_count*sizeof(float));
    float *sigmoidTerm = (float*)malloc(dataset.examples*sizeof(float));
    
    float *x_mul_theta = (float*)malloc(dataset.examples*sizeof(float));
    for(int i = 0; i < dataset.examples; i++){
       x_mul_theta[i] = 0;
    }
    
    for(int row = 0; row < dataset.examples; row++){
        for(int col = 0; col < dataset.feature_count; col++){
            x_mul_theta[row] += dataset.features[row][col]*theta[col];
        }
    }
    
    for(int i = 0; i < dataset.examples; i++){
        sigmoidTerm[i] = 1.0/(float)(1 + exp(-x_mul_theta[i])) - dataset.labels[i];
    }
    
    for(int col = 0; col < dataset.feature_count; col++){
        vect[col] = 0;
        for(int row = 0; row < dataset.examples; row++){
            vect[col] += dataset.features[row][col]*sigmoidTerm[row];
        }
        
    }
    
    return vect;
    
}

float* GradientDescentDFE(int streamSize, float* examples, float* theta, float* y){
    float *gradient = (float*)malloc(streamSize*vectorSize*sizeof(float));
    Vectors(streamSize, examples, theta, y, gradient);
    return gradient;
}

void FitModelDFE(struct Dataset dataset, float learningRate, int maxIter){
    int streamSize = dataset.examples;
    float *featuresSerialized = (float*)malloc(streamSize*vectorSize*sizeof(float));
    float* theta = (float*)malloc(streamSize*vectorSize*sizeof(float));
    int k = 0;
    for(int i = 0; i < streamSize; i++){
        for(int j = 0; j < vectorSize; j++){
            theta[vectorSize*i + j] = 0;
            if(i < dataset.examples && j < dataset.feature_count)
                featuresSerialized[k++] = dataset.features[i][j];
            else
                featuresSerialized[k++] = 0;
        }
    }
    
    
    for(int iter = 0; iter < maxIter; iter++){
        float* grad_partials = GradientDescentDFE(streamSize, featuresSerialized, theta, dataset.labels);
        float* grad = (float*)malloc(vectorSize*sizeof(float));
        for(int j = 0; j < vectorSize; j++){
            grad[j] = 0;
            for(int i = 0; i < streamSize; i++){
                grad[j] += grad_partials[i*vectorSize + j];
            }
        }
        
        for(int i = 0; i < vectorSize; i++){
            theta[i] -= learningRate*grad[i];
        }
        
        // Replicate theta
        for(int i = 1; i < streamSize; i++){
            for(int j = 0; j < vectorSize; j++){
                theta[vectorSize*i + j] = theta[j];
            }
        }
        
    }
    
    printf("Parametri:\n");
    for(int i = 0; i < dataset.feature_count; i++){
        printf("%f ", theta[i]);
    }
    
    
    
}

void FitModelCPU(struct Dataset dataset, float learningRate, int maxIter){
    float* theta = (float*)malloc(dataset.feature_count*sizeof(float));
    for(int i = 0; i < dataset.feature_count; i++){
        theta[i] = 0;
    }
    for(int iter = 0; iter < maxIter; iter++){
        float* grad = GradientDescentCPU(dataset, theta);
        for(int i = 0; i < dataset.feature_count; i++){
            theta[i] -= learningRate*grad[i];
        }
    }
    printf("Parametri:\n");
    for(int i = 0; i < dataset.feature_count; i++){
        printf("%f ", theta[i]);
    }
}

int main()
{
    struct Dataset dataset = AllocateMemory("dataset200.txt");
   
    ReadDataset("dataset200.txt", &dataset);
    printf("data_count: %d, features: %d\n", dataset.examples, dataset.feature_count);
    struct Dataset new_dataset = FeatureMap(dataset, 2);
    printf("-----Feature mapping (2nd degree)-----\n");
    printf("data_count: %d, features: %d\n", new_dataset.examples, new_dataset.feature_count);
  //  PrintDataset(new_dataset);
  //  printf("%s", "---------Labels----------------\n");
 //    for(int i = 0; i < new_dataset.examples; i++){
 //       printf("%f ", new_dataset.labels[i]);
 //   }
    printf("\n");
    printf("CPU:\n");
    FitModelCPU(new_dataset, 0.02, 500);
    printf("\nDFE:\n");
    FitModelDFE(new_dataset, 0.02, 500);
    printf("\nDone\n");
    return 0;
}
