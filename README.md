# OOPDeepLearningCPP; A horrible way to do deep learning
Udacity Capstone Project for the C++ nanodegree. In this project, I have implemented a multilayer perceptron for MNIST image classification. Contrary to normal solutions implemented with tensors and matrix algebra, in my solution each node is a unique object. As such, we miss the great great optimization we get from tensor operations and also lose the ability for hardware optimization with CUDA. Nevertheless, it's a fun project where we can really break down the math to individual numbers being updated and also allowing us to use fundamental C++ concepts required to pass the capstone project.

## Motivation (can be skipped)
My very first deep neural network was a multilayer perceptron implemented in similarly to this repo, but then in Python. Training a network was horribly slow as multiplications were done in vanilla python with for loops and such. Considering the increase in speed with a compiled language like C++, I was motivated to re-implement that old project as my capstone project.


## Building and Running
First, download this repo and unzip mnist_data.zip to get the MNIST dataset. If you are on the Udacity Workspace, you can use:
```
unzip mnist_data.zip
```

For compilation, I used the hello world makefile provided by Udacity to compile the program. You can perform the following steps to compile and run the program:

``` 
 mkdir build
 cd build
 cmake ..
 make
 ./OOPDeepLearning
 ```

When running the program there is an interaction between the program and the user. You will first be asked how many hidden layers you want to have. You will need to provide a number here. My recommendation would be 2.
Next, you are asked how many nodes you want per layer. I recommend 256 for the first hidden layer and 128 for the second hidden layer. Here a picture describing the entire process:

TODO: Add picture here

## Code Behavior & Class Structure

### Summary
The code creates a deep nearal network model based on input provided by the user. Both the forward pass and backward pass with backpropagation are implemented to learn the data. Over time, the model gets better at predicting the data, obtaining higher accuracy on the test set and lower loss on the training set.

### Classes overview

#### Node
During a forward pass, a node computes the sum of all nodes in the previous layer multiplied by some weights, which results in activation 'a'. Next, the node applied the ReLu activation funtion to get the node's output 'o'.

Backpropagation happens in two steps. First, each node computes its error term based on the error term and weights of the next layer. Second, the weights are updated in one Stochastic Gradient Descent (SGD) 'step'. Each node needs to know its error term and the output of the previous layer to do so.

**Private members/variables:**  
`generator`: Used to initialize the weights of each node. It is static as otherwise each node will have the same weights. The weights of the entire network need to be random.  
`_node_idx`: Index to identify the node in a layer.  
`_weights`: The weights between this node and all nodes of the previous layer. These weights are the parameters that the network updates to correctly predict the data.
`_prev_layer` and `_next_layer`: Pointers to the previous and next layer. Node pointers are used as information of nodes on previous layers change over time. Furthermore, by having a vector pointer, only two variables (the vector pointers) needs to be stored in a node. If it was a normal vector, each node would have to store as many node references as the previous and next layer have.  
`_a`: Node activation.  
`_o`: Node output.  
`_error`: The error term that is computed during backwards

**public functions:**  
`Node`: Initializer of Node. The index of this node in its layer is required as a parameter.  
`add_previous` and `add_next`: Sets a reference to the previous and next layer.  
`forward`: Computes the activation and output of this node.  
`backward`: Computes the error term of this node. If error is provided (required for the output layer), the node simply saves that error.  
`step`: Updates all weights of this node. The learning rate must be provided as a parameter.  
`get_...` and `set_...`: Used to retrieve or set the private member variables.  
Lastly, the copy contructor and copy assignment operator are set to delete as it is important that each node in the network has only one instance of the node class.

### Neural_net
The neural network class is the orchestrator of the deep learning process. It calls nodes in the correct sequence during the forward pass and backward pass. 

**Private variables and function.**  
`_lr`: The learning rate defining the step size for the nodes.  
`_add_layer`: A utility function used only by the neural net class itself to both create a new layer of nodes and store the references.  
`_layers`: All layers in the deep neural network. Nodes and layers (a vector of Nodes) are pointers as the number of required nodes and layers are provided by the user and therefore not known in advance.

**Public functions.**  
`NeuralNet`: Class contructor. The learning rate provided scales the gradients when the node weights are updated.  
`forward`: Call the nodes of all layers in sequence from input to output to perform a forward pass. This function expects an MNIST image as a vector of floats.  
`backward`: Calls the nodes of all but the input layer in sequence from output to input to compute the error terms. This function expects the gradient of the loss wrt. the network output as a vector of floats.  
`step`: Calls all nodes to update the weights.  
`print_network`: Utility function to print information about the network.  





For a better understanding, I can highly recommend [this](https://brilliant.org/wiki/backpropagation/) page of Brilliant: 

