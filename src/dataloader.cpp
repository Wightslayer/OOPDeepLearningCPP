#include <dirent.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

#include "dataloader.h"

using std::stof;
using std::string;
using std::to_string;
using std::vector;

void MNISTTrainLoader::next()
{
    // Prepare next sample
    _index = _index_distr(_gen);
}

void MNISTTestLoader::next()
{
    // Prepare next sample
    _index = (_index + 1) % 10000;
    
}

vector<float> MNISTDataloader::get_image()
{
    // Returns the next image with pixel values between -1 and 1
    vector<float> image = _images[_index];
    vector<float> normalised_image;


    for (float pixel : image)
    {
        normalised_image.push_back(pixel / 128 - 1.0);  // between -1 and 1
    }

    return normalised_image;
}

vector<int> MNISTDataloader::get_label()
{
    // Get the next label in one-hot format
    vector<int> label;

    for (int i = 0; i < 10; i++)
    {
        if (i == _labels[_index])
        {
            label.push_back(1);
        }else
        {
            label.push_back(0);
        }

    }

    return label;
}


vector<int> MNISTDataloader::_get_labels(string path)
{
    string line;
    string label;
    vector<int> labels;
    std::istringstream linestream;


    std::ifstream filestream(path);

    if (filestream.is_open()) {
        while(std::getline(filestream, line)){
            std::istringstream linestream(line);
            linestream >> label;
            labels.push_back(std::stoi(label));
        }        
    }

    return labels;
}

vector<vector<float>> MNISTDataloader::_get_images(string path)
{
    string line;
    string pixel;
    vector<vector<float>> images;

    std::ifstream filestream(path);

    if (filestream.is_open()) {
        while(std::getline(filestream, line)){
            vector<float> image;
            std::istringstream linestream(line);
            while (linestream >> pixel)
            {
                image.push_back(std::stof(pixel));
            }
            images.push_back(image);
        }        
    }

    return images;
}
    
MNISTTrainLoader::MNISTTrainLoader()
{
    _gen = std::mt19937(rd());
    _index_distr = std::uniform_int_distribution<int>(0, 59999);
    _index = _index_distr(_gen);

    _images = _get_images("../mnist_txt/train_images.txt");
    _labels = _get_labels("../mnist_txt/train_labels.txt");
}


MNISTTestLoader::MNISTTestLoader()
{
    _index = 0;

    _images = _get_images("../mnist_txt/test_images.txt");
    _labels = _get_labels("../mnist_txt/test_labels.txt");
}

void MNISTDataloader::draw_image(vector<float> image)
{
    float pixel;
    // Image is 28 x 28 pixels. 28 * 28 = 784.
    for (int i = 0; i < 784; i++){
        pixel = image[i];

        if (pixel > 0){
            std::cout << "# ";
        }else{
            std::cout << ". ";
        }

        if ((i+1) % 28 == 0){
            std::cout << "\n";
        }
    }
}


void MNISTDataloader::draw_image_detail(vector<float> image)
{
    float pixel;
    // Image is 28 x 28 pixels. 28 * 28 = 784.
    for (int i = 0; i < 784; i++){
        pixel = image[i];

        std::cout << pixel << " ";
        if (pixel < 100){
            std::cout << " ";
        }
        if (pixel < 10){
            std::cout << " ";
        }

        if ((i+1) % 28 == 0){
            std::cout << "\n";
        }
    }
}

// int main()
// {
//     // MNISTTrainLoader train_loader;
//     MNISTTrainLoader test_loader;
//     int label;
//     vector<float> image;
//     label = test_loader.get_label();
//     image = test_loader.get_image();

//     std::cout << label << "\n";
//     test_loader.draw_image(image);

//     test_loader.next();
//     label = test_loader.get_label();
//     image = test_loader.get_image();

//     std::cout << label << "\n";
//     test_loader.draw_image(image);

//     test_loader.next();
//     label = test_loader.get_label();
//     image = test_loader.get_image();

//     std::cout << label << "\n";
//     test_loader.draw_image(image);

//     test_loader.next();
//     label = test_loader.get_label();
//     image = test_loader.get_image();

//     std::cout << label << "\n";
//     test_loader.draw_image(image);
    

//     return 0;
// }

