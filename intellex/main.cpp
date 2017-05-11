#include <iostream>
#include "Network.hpp"

int main(int argc, const char * argv[]) {
    /* e.g. { 3, 2, 1 } */
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Network myNet(topology);
    
    vector<double> in, targetVals, out;
    /* now, use your imagination ... */
}
