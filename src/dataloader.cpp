#include <sstream>
#include <fstream>

#include "dataloader.h"

using std::string;
using std::vector;

void MNISTTrainLoader::next()
{
    // Generate index for next sample
    _index = _index_distr(_gen);
}

void MNISTTestLoader::next()
{
    // Compute index for next sample
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
        // One-hot. One if label (0,1,...,9) equals index i, zero otherwise.
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
    // Reads all labels from a MNIST_txt dataset split.
    // Path is the path to the labels file
    // Code modified from Udacity's system monitor assessment

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

    return labels;  // All labels are stored in memory
}

vector<vector<float>> MNISTDataloader::_get_images(string path)
{
    // Reads all images from a MNIST_txt dataset split.
    // Path is the path to the images file
    // Code modified from Udacity's system monitor assessment

    string line;
    string pixel;
    vector<vector<float>> images;  // All images are stored in memory

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
    // Constructs the MNISTTrainLoader class.
    
    // Sample training samples randomly
    _gen = std::mt19937(rd());
    _index_distr = std::uniform_int_distribution<int>(0, 59999);
    _index = _index_distr(_gen);

    _images = _get_images("../mnist_txt/train_images.txt");
    _labels = _get_labels("../mnist_txt/train_labels.txt");
}


MNISTTestLoader::MNISTTestLoader()
{
    // Constructs the MNISTTestLoader class.
    
    // Sample testing samples in order  
    _index = 0;

    _images = _get_images("../mnist_txt/test_images.txt");
    _labels = _get_labels("../mnist_txt/test_labels.txt");
}

void MNISTDataloader::draw_image(vector<float> image)
{
    // Visualizes the image on the terminal in ascii format
    // The output has a '#' for non-zero pixels and '.' otherwise.

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
    // Visualizes the image on the terminal with each pixel being printed as the actual number.

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
