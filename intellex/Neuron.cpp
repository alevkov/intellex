#include "Neuron.hpp"

double Neuron::eta = 0.15; /* rate of learning */
double Neuron::alpha = 0.5; /* momentum of learning */

Neuron::Neuron(unsigned numOutputs, unsigned idx)
{
    __idx = idx;
    for (int c = 0; c < numOutputs; ++c) {
        __outputWeights.push_back(Edge());
        __outputWeights.back().weight = Neuron::randomWeight();
    }
}

double Neuron::sumDerivativesOfWeights(const Layer & next)
{
    double sum = 0.0;
    
    for (unsigned n = 0; n < next.size() - 1; ++n) {
        sum += __outputWeights[n].weight * next[n].__gradient;
    }
    
    return sum;
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - __outputVal;
    __gradient = delta * Neuron::transferFuncDerivative(__outputVal);
}

void Neuron::calcHiddenGradients(const Layer & next)
{
    double dow = sumDerivativesOfWeights(next);
    __gradient = dow * Neuron::transferFuncDerivative(__outputVal);
}

void Neuron::updateInputWeights(Layer & prev)
{
    for (unsigned n = 0; n < prev.size(); ++n) {
        Neuron &neuron = prev[n];
        double oldDeltaWeight = neuron.__outputWeights[__idx].deltaWeight;
        double newDeltaWeight =
        /* individual input, magnified by the gradient and train rate: */
        eta /* learning rate */
        * neuron.getOutputVal()
        * __gradient
        /* also add momentum = a fraction of the previous delta weight */
        + alpha
        * oldDeltaWeight;
        neuron.__outputWeights[__idx].deltaWeight = newDeltaWeight;
        neuron.__outputWeights[__idx].weight += newDeltaWeight;
    }
}

void Neuron::feedForward(const Layer & prev)
{
    double sum = 0.0;
    
    /*
     * sum the previous output values with weight multiplied
     * output = f(∑ v[i] * w[i])
     */
    for (unsigned n = 0; n < prev.size(); ++n) {
        sum += prev[n].__outputVal *
        prev[n].__outputWeights[__idx].weight;
    }
    
    __outputVal = Neuron::transferFunc(sum);
}

/*
 * Activation Function
 */

double Neuron::transferFunc(double x)
{
    return tanh(x);
}

double Neuron::transferFuncDerivative(double x)
{
    return 1 - (tanh(x) * tanh(x));
}
