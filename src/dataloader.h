#ifndef DATALOADER_H
#define DATALOADER_H

#include <dirent.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>

using std::stof;
using std::string;
using std::to_string;
using std::vector;


class MNISTDataloader{

    public:
    void next();
    vector<float> get_image();
    vector<int> get_label();
    void draw_image(vector<float>);
    void draw_image_detail(vector<float>);

    protected:
    vector<vector<float>> _images;
    vector<int> _labels;
    int _index;
    std::default_random_engine _generator;

    vector<int> _get_labels(string path);
    vector<vector<float>> _get_images(string path);
};


class MNISTTrainLoader : public MNISTDataloader{
    public:
    MNISTTrainLoader();

    void next();  // Overload

    private:
    // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
    std::random_device rd;
    std::mt19937 _gen;
    std::uniform_int_distribution<> _index_distr;
};

class MNISTTestLoader : public MNISTDataloader{
    
    public:
    MNISTTestLoader();

    void next();  // Overload
};

#endif