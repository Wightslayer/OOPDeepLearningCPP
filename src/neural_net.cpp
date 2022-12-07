#include <vector>
#include <iostream>
#include <string>
#include <memory>

#include "neural_net.h"

using std::vector;
using std::cout;
using std::cin;
using std::string;

NeuralNet::NeuralNet(float lr)
{
    _lr = lr;  // Set the learning rate

    int input_nodes = 784;  // 28 * 28
    int output_nodes = 10;  // 10 digits
    int n_layers;
    int n_nodes;
    vector<Node *> layer;

    Node *n;

    string user_input;

    _add_layer(input_nodes, true);

    cout << "How many layers would you like to add?\n";
    cin >> user_input;
    n_layers = std::stoi(user_input);
    for (int i = 1; i < n_layers + 1; i++)
    {
        cout << "How many nodes for layer " << i << "?\n";
        cin >> user_input;
        n_nodes = std::stoi(user_input);
        _add_layer(n_nodes, true);
    }

    _add_layer(output_nodes, false);
}

vector<float> NeuralNet::forward(vector<float> image)
{
    int n_layers = _layers.size();
    int n_nodes;
    float pixel;
    vector<Node *> *layer;
    vector<Node *> *input_layer;

    vector<float> output;
    
    input_layer = _layers[0];
    for (int i = 0; i < 784; i++)  // 28 * 28
    {
        pixel = image[i];
        (*input_layer)[i]->set_output(pixel);
    }

    for (int i = 1; i < n_layers; i++)
    {
        layer = _layers[i];

        // Call forward on all but last node, as that is the bias node.
        n_nodes = layer->size();
        for (int j = 0; j < n_nodes -1; j++)
        {
            (*layer)[j]->forward();
        }
    }

    for (int j = 0; j < n_nodes -1; j++)
    {
        output.push_back((*layer)[j]->get_activation());
    }

    return output;
}

void NeuralNet::backward(vector<float> grads)
{
    vector<Node *> *layer = _layers.back();
    for (int i = 0; i < layer->size(); i++)
    {
        (*layer)[i]->backward(grads[i]);
    }

    for (int i = _layers.size() - 2; i > 0; i--)
    {
        layer = _layers[i];
        for (Node *n : *layer)
        {
            n->backward();
        }
    }
}

void NeuralNet::step()
{
    for (vector<Node *> *layer : _layers)
    {
        for (Node *n : *layer)
        {
            n->step(_lr);
        }
    }
}


void NeuralNet::_add_layer(int n_nodes, bool bias)
{
    Node *n;
    vector<Node *> *layer = new vector<Node *>;
    int n_layers = _layers.size();

    for (int i = 0; i < n_nodes; i++)
    {
        n = new Node(i);  // TODO unique_ptr
        if (n_layers > 0)
        {
            n->add_previous(_layers.back());
        }
        layer->push_back(n);
    }

    if (bias)
    {
        n = new Node(n_nodes);  // TODO unique_ptr
        n->set_output(0.0);  // Bias node
        layer->push_back(n);
    }

    if (n_layers > 1)
    {
        vector<Node *> *prev_layer = _layers.back();
        for (Node *n : *prev_layer)
        {
            n->add_next(layer);
        }
    }

    _layers.push_back(layer);
}


void NeuralNet::print_network()
{
    int n_layers = _layers.size();
    int n_params = 0;


    for (int i = 1; i < n_layers; i++)
    {
        n_params += _layers[i-1]->size() * _layers[i]->size();
    }

    cout << "#Layers: " << n_layers << "\n";
    cout << "#Parameters: " << n_params << "\n";
    cout << "Network architecture (includes bias):\n";
    for (int i = 0; i < n_layers; i++)
    {
        cout << _layers[i]->size();
        if (i == 0)
        {
            cout << " (input) --> ReLu --> ";
        } else if (i == n_layers - 1)
        {
            cout << " (output)\n";
        }else{
            cout << " --> ReLu --> ";
        }
    }
}
