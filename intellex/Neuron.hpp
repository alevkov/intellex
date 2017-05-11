#ifndef Neuron_hpp
#define Neuron_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#define INPUT_LAYER 0

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

struct Edge
{
    double weight;
    double deltaWeight;
};

class Neuron
{
 public:
    Neuron(unsigned numOutputs, unsigned idx);
    ~Neuron() { };
    void setOutputVal(double val) { this->__outputVal = val; }
    double getOutputVal(void) const { return __outputVal; }
    void feedForward(const Layer & prev);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer & next);
    void updateInputWeights(Layer & prev);
    
 private:
    static double eta; /* [0.0 ... 1.0] training rate */
    static double alpha; /* [0.0 ... 1.0] momentum */
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    static double transferFunc(double x);
    static double transferFuncDerivative(double x);
    double sumDerivativesOfWeights(const Layer & next);
    double __outputVal;
    double __gradient;
    unsigned __idx;
    vector<Edge> __outputWeights;
};

#endif /* Neuron_hpp */
