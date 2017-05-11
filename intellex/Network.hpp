#ifndef Network_hpp
#define Network_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "Neuron.hpp"

using namespace std;

class Network
{
 public:
    Network(const vector<unsigned> & topology);
    ~Network() { };
    void feedForward(const vector<double> & in);
    void backProp(const vector<double> & targetVals);
    void getResults(vector<double> & out);
 private:
    vector<Layer> __layers;
    double __err;
    double __recentAverageErr;
    double __recentAverageSmoothingFactor;
};

#endif /* Network_hpp */
