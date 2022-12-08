#include <vector>
#include <iostream>
#include <random>
#include <math.h>

#include "node.h"
using std::vector;

float ReLu(float a)
{
    // ReLu activation function
    if (a > 0)
    {
        return a;
    }else
    {
        return 0;
    }
}

Node::Node(int idx)
{
    // Index of this node in its layer
    _node_idx = idx;
}

void Node::add_previous(vector<Node *> *layer)
{
    // Add a reference to the previous layer and initialises weights to each node.
    _prev_layer = layer;
    int n_nodes = _prev_layer->size();
    float weight;

    // he initialization: https://arxiv.org/abs/1502.01852
    // Also: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks
    // Modified from the C++ website: https://cplusplus.com/reference/random/normal_distribution/
    std::normal_distribution<float> weight_sampler = std::normal_distribution<float>(0.0, sqrt(2.0 / n_nodes));

    // All but the bias node
    for (int i = 0; i < n_nodes - 1; i++)
    {
        weight = weight_sampler(generator);
        _weights.push_back(weight);
    }

    // The bias node
    _weights.push_back(0.0); 
}

void Node::add_next(vector<Node *> *layer)
{
    // Add reference to next layer
    _next_layer = layer;
}

void Node::forward()
{
    // Forward pass of just this node

    // Compute activation of this node
    _a = 0;  // Value before activation function
    for (int i = 0; i < _prev_layer->size(); i++)
    {
        _a += _weights[i] * (*_prev_layer)[i]->get_output();
    }

    // Compute output of this node
    _o = ReLu(_a);
}

void Node::backward()
{
    // First part of backpropagation. Compute the error term

    _error = 0;
    Node *n;
    if (_a > 0)  // If a <= 0, the gradient wrt. RELU is 0. 0 * error = 0
    {
        for (int i = 0; i < (*_next_layer).size() - 1; i++)
        {
            n = (*_next_layer)[i];
            _error += n->get_weight(_node_idx) * n->get_error();
        }
    }
}

void Node::backward(float error)
{
    // First part of backpropagation for the output layer

    _error = error;
}

void Node::step(float lr)
{
    // Second part of backpropagation. Update the weights

    for (int i = 0; i < _weights.size(); i++)
    {
        _weights[i] -= lr * (_error * (*_prev_layer)[i]->get_activation());
    }
}

// Getters and setters.
float Node::get_weight(int idx) {return _weights[idx];}  // The weight to node 'idx' of the previous layer
float Node::get_error(){return _error;}
float Node::get_output() {return _o;}
void Node::set_output(float o){_o = o;}

float Node::get_activation() {return _a;}
