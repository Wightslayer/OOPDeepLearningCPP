
#ifndef NODE_H
#define NODE_H

#include <vector>
#include <random>

using std::vector;

class Node
{
    public:
    Node(int idx);

    void add_previous(vector<Node *> *layer);
    void add_next(vector<Node *> *layer);

    // Deep learning magic
    void forward();
    void backward();  // Backward only propagates the error term, not updates the weights
    void backward(float error);  // The output layer receives the error term, as there are no further layers to get it from
    void step(float lr);  // Actually update the weights here

    // Getters
    float get_weight(int idx);  // Return the weight to node idx of the previous layer
    float get_error();
    float get_activation();
    float get_output();

    // Setters
    void set_output(float o);  // Used to prepare input layer

    // Node should not be copied!
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    private:
    // Modified from the C++ website: https://cplusplus.com/reference/random/normal_distribution/
    static std::default_random_engine _generator;  // Make static to prevent all nodes from having same weights

    int _node_idx;  // Index of the node in a layer.
    vector<float> _weights;  // One weight per node in the previous layer (includes bias)
    vector<Node *> *_prev_layer;  // Pointer to previous layer
    vector<Node *> *_next_layer;  // Pointer to next layer
    float _a;  // Activation
    float _o;  // Output
    float _error;  // The error term of this node, for next (previous) layer during backprop.

};

// We have to initialize the variable so it can be used in the nodes in the network.
// This stackoverflow post describes why we do the line below.
// https://stackoverflow.com/questions/37053677/initializing-static-default-random-engine
std::default_random_engine Node::_generator = std::default_random_engine{};

#endif
