#include <vector>
#include <iostream>
#include <random>
#include <math.h>

#include "node.h"
using std::vector;

float ReLu(float a)
{
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
    _node_idx = idx;
}

void Node::add_previous(vector<Node *> *layer)
{
    _prev_layer = layer;
    int n_nodes = _prev_layer->size();
    float weight;

    // he initialization: https://arxiv.org/abs/1502.01852
    // Also: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks
    // Modified from the C++ website: https://cplusplus.com/reference/random/normal_distribution/
    std::normal_distribution<float> weight_sampler = std::normal_distribution<float>(0.0, sqrt(2.0 / n_nodes));

    for (int i = 0; i < n_nodes; i++)
    {
        weight = weight_sampler(generator);
        _weights.push_back(weight);
    }
}

void Node::add_next(vector<Node *> *layer)
{
    _next_layer = layer;
}

void Node::forward()
{
    _a = 0;  // Value before activation function
    for (int i = 0; i < _prev_layer->size(); i++)
    {
        _a += _weights[i] * (*_prev_layer)[i]->get_output();
    }
    _o = ReLu(_a);
}

void Node::backward()
{
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
    _error = error;
}

void Node::step(float lr)
{
    for (int i = 0; i < _weights.size(); i++)
    {
        _weights[i] -= lr * (_error * (*_prev_layer)[i]->get_activation());
    }
}

float Node::get_weight(int idx) {return _weights[idx];}
float Node::get_error(){return _error;}
float Node::get_output() {return _o;}
void Node::set_output(float o){_o = o;}

float Node::get_activation() {return _a;}
