#include "Network.hpp"

Network::Network(const vector<unsigned> & topology)
{
    size_t numLayers = topology.size();
    /* i -> layer number */
    for (int layer = 0; layer < numLayers; ++layer) {
        __layers.push_back(Layer());
        /* add individual neurons */
        unsigned numOutputs = layer == topology.size() - 1 ? 0 : topology[layer + 1];
        
        for (int neuron = 0; neuron <= topology[layer]; ++neuron) {
            __layers.back().push_back(Neuron(numOutputs, neuron));
            cout << "created a neuron" << endl;
        }
        
        __layers.back().back().setOutputVal(1.0);
    }
}

void Network::feedForward(const vector<double> & in)
{
    assert(in.size() == __layers[INPUT_LAYER].size() - 1);
    
    for (unsigned i = 0; i < in.size(); ++i) {
        __layers[INPUT_LAYER][i].setOutputVal(in[i]);
    }
    
    /* forward propagate */
    
    for (unsigned layer = 1; layer < __layers.size(); ++layer) {
        Layer &prev = __layers[layer - 1];
        for (unsigned n = 0; n < __layers[layer].size() - 1; ++n) {
            __layers[layer][n].feedForward(prev);
        }
    }
}

void Network::backProp(const vector<double> & targetVals)
{
    Layer &out = __layers.back();
    __err = 0.0; /* root mean squared error */
    
    for (unsigned n = 0; n < out.size() - 1; ++n) {
        double delta = targetVals[n] - out[n].getOutputVal();
        __err = delta * delta;
    }
    
    __err /= out.size() - 1; /* get average error squared */
    __err = sqrt(__err);
    
    /* implement a recent average measurement */
    
    __recentAverageErr =
    (__recentAverageErr * __recentAverageSmoothingFactor + __err)
    / (__recentAverageSmoothingFactor + 1.0);
    
    /* calculate output layer gradients */
    
    for (unsigned n = 0; n < out.size() - 1; ++n) {
        out[n].calcOutputGradients(targetVals[n]);
    }
    
    /* calculate gradients on hidden layers */
    
    for (size_t layer = __layers.size() - 2; layer > 0 ; --layer) {
        Layer &hidden = __layers[layer];
        Layer &next = __layers[layer + 1];
        
        for (unsigned n = 0; n < hidden.size(); ++n) {
            hidden[n].calcHiddenGradients(next);
        }
    }
    
    /* calculate weights */
    
    for (size_t layer = __layers.size() - 1; layer > 0; --layer) {
        Layer &l = __layers[layer];
        Layer &prev = __layers[layer - 1];
        
        for (unsigned n = 0; n < l.size(); ++n) {
            l[n].updateInputWeights(prev);
        }
    }
}

void Network::getResults(vector<double> & out)
{
    out.clear();
    
    for (unsigned n = 0; n < __layers.back().size() - 1; ++n) {
        out.push_back(__layers.back()[n].getOutputVal());
    }
}
