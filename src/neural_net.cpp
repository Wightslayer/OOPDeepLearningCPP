#include <vector>
#include <iostream>
#include <string>

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

    string user_input;  // For the use input

    _add_layer(input_nodes, true);  // Input layer

    // Get the number of layers to add from the user
    cout << "How many layers would you like to add?\n";
    cin >> user_input;
    n_layers = std::stoi(user_input);
    for (int i = 1; i < n_layers + 1; i++)
    {
        // Get the number of nodes in layer 'i'. 
        cout << "How many nodes for layer " << i << "?\n";
        cin >> user_input;
        n_nodes = std::stoi(user_input);
        _add_layer(n_nodes, true);  // Add the layer with the number of nodes as specified by the user
    }

    _add_layer(output_nodes, false);  // Output layer
}

vector<float> NeuralNet::forward(vector<float> image)
{
    // Perform the forward pass

    int n_layers = _layers.size();
    int n_nodes;
    float pixel;
    vector<Node *> *layer;  // Pointer to a layer
    vector<Node *> *input_layer;  // Pointer to the input layer

    vector<float> output;
    
    // Set the output of the input layer with the pixel values of the image
    input_layer = _layers[0];
    for (int i = 0; i < 784; i++)  // 28 * 28
    {
        pixel = image[i];
        (*input_layer)[i]->set_output(pixel);
    }

    // Forward all but input layer.
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

    // Get the output of the network
    for (int j = 0; j < n_nodes -1; j++)
    {
        output.push_back((*layer)[j]->get_activation());
    }

    // Return the output of the network
    return output;
}

void NeuralNet::backward(vector<float> grads)
{
    // First part of backpropagation. Computes the error term of the nodes

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
    // Second part of backpropagation. Update all the weights of all nodes.

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
    // Utility function to add a layer to the neural network.

    Node *n;
    vector<Node *> *layer = new vector<Node *>;  // On dynamic memory
    int n_layers = _layers.size();

    for (int i = 0; i < n_nodes; i++)
    {
        n = new Node(i); 
        if (n_layers > 0)  // Input layer has no previous layer
        {
            n->add_previous(_layers.back());
        }
        layer->push_back(n);
    }

    if (bias)  // Trick to make bias just another node
    {
        n = new Node(n_nodes);  
        n->set_output(1.0);  // Bias node
        layer->push_back(n);
    }

    // If this is not the input layer, give the layer a pointer to the just made layer
    if (n_layers > 1)
    {
        vector<Node *> *prev_layer = _layers.back();
        for (Node *n : *prev_layer)
        {
            n->add_next(layer);
        }
    }

    _layers.push_back(layer);  // Add layer to the neural network
}


void NeuralNet::print_network()
{
    // Utility function to print network information

    int n_layers = _layers.size();
    int n_params = 0;

    // Compute number of parameters
    for (int i = 1; i < n_layers; i++)
    {
        n_params += _layers[i-1]->size() * _layers[i]->size();
    }

    cout << "#Layers: " << n_layers << "\n";
    cout << "#Parameters: " << n_params << "\n";

    // Print the network overview to terminal
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
