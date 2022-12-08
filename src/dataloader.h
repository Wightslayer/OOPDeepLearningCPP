#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>

using std::stof;
using std::string;
using std::to_string;
using std::vector;


class MNISTDataloader{

    public:
    void next();  // Determine next sample index
    vector<float> get_image();  // Returns image at index
    vector<int> get_label();  // return label at index
    void draw_image(vector<float>);  // Prints image in terminal
    void draw_image_detail(vector<float>);  // Prints image in terminal

    protected:
    vector<vector<float>> _images;  // All images
    vector<int> _labels;  // All labels
    int _index;  // Index of current image and label

    vector<int> _get_labels(string path);  // Reads labels from disk
    vector<vector<float>> _get_images(string path);  // Reads images from disk
};


class MNISTTrainLoader : public MNISTDataloader{
    public:
    MNISTTrainLoader();

    void next();  // Overload

    private:
    // Like Udacity's traffic light assignment.
    // Also: https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
    std::random_device rd;
    std::mt19937 _gen;
    std::uniform_int_distribution<> _index_distr;
    std::default_random_engine _generator;
};

class MNISTTestLoader : public MNISTDataloader{
    
    public:
    MNISTTestLoader();

    void next();  // Overload
};

#endif