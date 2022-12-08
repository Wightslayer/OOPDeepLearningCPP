# OOPDeepLearningCPP; A horrible way to do deep learning
Udacity Capstone Project for the C++ nanodegree. In this project, I have implemented a multilayer perceptron for MNIST image classification. Contrary to normal solutions implemented with tensors and matrix algebra, in my solution each node is a unique object. As such, we miss the great great optimization we get from tensor operations and also lose the ability for hardware optimization with CUDA. Nevertheless, it's a fun project where we can really break down the math to individual numbers being updated and also allowing us to use fundamental C++ concepts required to pass the capstone project.

## Motivation (can be skipped)
My very first deep neural network was a multilayer perceptron implemented in similarly to this repo, but then in Python. Training a network was horribly slow as multiplications were done in vanilla python with for loops and such. Considering the increase in speed with a compiled language like C++, I was motivated to re-implement that old project as my capstone project.


## Building and Running
First, download this repo and unzip mnist_data.zip to get the MNIST dataset.

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





For a better understanding, I can highly recommend [this](https://brilliant.org/wiki/backpropagation/) page of Brilliant: 

