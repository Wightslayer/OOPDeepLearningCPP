#include <iostream>
#include <vector>
#include <math.h>

#include "node.cpp"
#include "neural_net.cpp"
#include "dataloader.cpp"

using std::vector;


void SoftMaxGradLoss(vector<float> logits, vector<int> target, float &loss, vector<float> &gradient)
{
    // Computes the softmax loss and gradient of the logits wrt. the target. 
    // The loss and gradient are provided by reference to provide more than one output with this function

    float exp_result;
    vector<float> exp_results;
    float probability;

    // Prevent softmax overflow: https://stats.stackexchange.com/questions/304758/softmax-overflow
    float max = -1;
    for (float logit : logits)
    {
        if (logit > max)
        {
            max = logit;
        }
    }

    // Prepare to compute gradient and loss
    float expsum = 0;
    for (float logit : logits)
    {
        exp_result = exp(logit - max);
        exp_results.push_back(exp_result);
        expsum += exp_result;
    }

    // Compute the loss and gradient
    for (int i = 0; i < logits.size(); i++)
    {
        probability = exp_results[i] / expsum;
        gradient.push_back(probability - target[i]);
        if (target[i] == 1)
        {
            loss = -(log(probability));  // Natural Logarithm
        }
    }
}

void eval_model(MNISTTestLoader dataloader, NeuralNet model)
{
    // Evaluates the model on the MNIST test set

    vector<float> image;
    vector<int> target;
    vector<float> pred;
    float max_prob = 0;
    int max_ind = 0;
    int correct = 0;

    std::cout << "Evaluating model performance...\n";
    for (int i = 0; i < 10000; i++)  // For each image in test set
    {
        image = dataloader.get_image();
        target = dataloader.get_label();

        pred = model.forward(image);

        // Find predicted number
        for (int j = 0; j < 10; j++)
        {
            if (pred[j] > max_prob)
            {
                max_prob = pred[j];
                max_ind = j;
            }
        }
        max_prob = 0;

        // Keep track of number of correct predictions
        if (target[max_ind] == 1)
        {
            correct++;
        }


        dataloader.next();
    }

    std::cout << "Test accuracy: " << (float)correct / 10000 * 100 << "%\n";
}

int main(){

    float learning_rate = 1e-6;
    int eval_every = 10000;

    NeuralNet nnet(learning_rate);

    MNISTTrainLoader train_loader;
    MNISTTestLoader test_loader;
    
    vector<float> image;
    vector<int> target;
    vector<float> output;
    vector<float> gradient;

    float loss;
    float avg_loss = 0;
    int epoch = 0;

    std::cout << "\n";
    nnet.print_network();
    std::cout << "\n";

    eval_model(test_loader, nnet);

    for (int i = 1; 1 < 100000000; i++)
    {
        // Prints for user
        if ((i - 1) % 60000 == 0)
        {
            epoch++;
            std::cout << " \n";
            std::cout << ">>>>> Epoch " << epoch << ".\n";
        }

        // Get next training sample
        train_loader.next();
        image = train_loader.get_image();
        target = train_loader.get_label();

        // Learn the data
        gradient.clear();
        output = nnet.forward(image);
        SoftMaxGradLoss(output, target, loss, gradient);
        nnet.backward(gradient);
        nnet.step();

        // prints for user
        avg_loss += loss;
        if (i % eval_every == 0)
        {
            avg_loss = avg_loss / eval_every;
            
            std::cout << "\n";
            eval_model(test_loader, nnet);
            std::cout << "Train loss: " << avg_loss << "\n";

            avg_loss = 0;
        }
        
    }

    return 0;
}
