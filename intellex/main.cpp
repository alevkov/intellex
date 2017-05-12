#include <iostream>
#include "Network.hpp"

int main(int argc, const char * argv[]) {

    vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);
    Network myNet(topology);
    
    vector<vector<double>> in, targetVals;
    
    /* training net to act as an OR gate */
    vector<double>in0 = {0, 1};
    vector<double>in1 = {0, 0};
    vector<double>in2 = {1, 0};
    vector<double>in3 = {1, 1};
    
    in.push_back(in0);
    in.push_back(in1);
    in.push_back(in2);
    in.push_back(in3);
    
    vector<double>t0 = {1};
    vector<double>t1 = {0};
    vector<double>t2 = {1};
    vector<double>t3 = {1};
    
    targetVals.push_back(t0);
    targetVals.push_back(t1);
    targetVals.push_back(t2);
    targetVals.push_back(t3);
    
    for (int i = 0; i < 90000; i++) {
        cout << "Pass " << i << endl;
        for (int j = 0; j <= 3; j++) {
            myNet.feedForward(in[j]);
            cout << "Input: " << in[j][0] << " " << in[j][1] << endl;
            myNet.backProp(targetVals[j]);
            vector<double> out;
            myNet.getResults(out);
            cout << "Output: " << out[0] << endl;
            cout << "Average error: " << myNet.getAvgError() << endl;
        }
        cout << endl;
    }
}
