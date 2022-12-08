#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>

#include "node.h"

using std::vector;

// Through trail and error, I found out that vectors move their data across memory adresses.
// This link confirms it also, which is why we have a vector of vector pointers for layers
// https://stackoverflow.com/questions/2447392/does-stdvector-change-its-address-how-to-avoid

class NeuralNet
{
    public:
    NeuralNet(float lr);  // Learning rate must be provided

    vector<float> forward(vector<float>);  // Forward pass
    void backward(vector<float> grads);  // Gradients of loss wrt. network output
    void step();  // Updates the weights of the neural network

    void print_network();  // print network information

    private:
    float _lr;
    void _add_layer(int n_nodes, bool bias);
    vector<vector<Node *>*> _layers;  // The neural network
};


#endif