#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include "node.cpp"
#include "neural_net.cpp"
#include "dataloader.cpp"

using std::vector;


void SoftMaxLoss(vector<float> logits, vector<int> target, float &loss, vector<float> &gradient)
{
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

    float expsum = 0;
    for (float logit : logits)
    {
        exp_result = exp(logit - max);
        exp_results.push_back(exp_result);
        expsum += exp_result;
    }

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
    vector<float> image;
    vector<int>  target;
    vector<float> pred;
    float max_prob = 0;
    int max_ind = 0;
    int correct = 0;

    std::cout << "Evaluating model performance...\n";
    for (int i = 0; i < 10000; i++)
    {
        image = dataloader.get_image();
        target = dataloader.get_label();

        pred = model.forward(image);

        for (int j = 0; j < 10; j++)
        {
            if (pred[j] > max_prob)
            {
                max_prob = pred[j];
                max_ind = j;
            }
        }
        max_prob = 0;
        if (target[max_ind] == 1)
        {
            correct++;
        }


        dataloader.next();
    }

    std::cout << "Test accuracy: " << (float)correct / 10000 * 100 << "%\n";
}

int main(){

    float learning_rate = 1e-5;

    NeuralNet nnet(learning_rate);

    vector<float> gradient;

    MNISTTrainLoader train_loader;
    MNISTTestLoader test_loader;
    vector<float> image;
    vector<int>  target;
    vector<float> output;

    int eval_every = 10000;
    float loss;
    float avg_loss = 0;
    int epoch = 0;

    std::cout << "\n";
    nnet.print_network();
    std::cout << "\n";

    eval_model(test_loader, nnet);

    for (int i = 1; i < 100000; i++)
    {
        if ((i - 1) % 60000 == 0)
        {
            epoch++;
            std::cout << " \n";
            std::cout << ">>>>> Epoch " << epoch << ".\n";
        }

        train_loader.next();
        image = train_loader.get_image();
        target = train_loader.get_label();

        gradient.clear();
        output = nnet.forward(image);
        SoftMaxLoss(output, target, loss, gradient);
        nnet.backward(gradient);
        nnet.step();

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