#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    // Google logging needed for parts of caffe that were extracted

    // Network structure
    // data - conv - reLU - pool - fc - softmax

    std::vector<Layer*> network;

    // Description of the neural network

    int N = 100; // number of samples/batch_size
    int d_w = 32; // data width
    int d_h = 32; // data height
    int ch = 3; // number of channels

    Image<float> data(32, 32, 3, 100);
    DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
    network.push_back(d_layer);
    printf("data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
                                                d_layer->out_dim_size(1),
                                                d_layer->out_dim_size(2),
                                                d_layer->out_dim_size(3));
    int n_f = 32; // number of filters
    int f_w = 7;  // filter width
    int f_h = 7;  // filter height
    int pad = (f_w-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter evaluated

    Convolutional * conv  = new Convolutional(n_f, f_w, f_h, pad,
                                              stride, d_layer);
    network.push_back(conv);
    printf("conv out size %d x %d x %d x %d\n", conv->out_dim_size(0),
                                                conv->out_dim_size(1),
                                                conv->out_dim_size(2),
                                                conv->out_dim_size(3));

    ReLU * relu = new ReLU(conv);
    network.push_back(relu);

    int p_w = 2; // pooling width
    int p_h = 2; // pooling height
    int p_stride = 2; // pooling stride

    MaxPooling * pool = new MaxPooling(p_w, p_h, p_stride, relu);
    network.push_back(pool);
    printf("pool out size %d x %d x %d x %d\n", pool->out_dim_size(0),
                                                pool->out_dim_size(1),
                                                pool->out_dim_size(2),
                                                pool->out_dim_size(3));

    Flatten * flatten = new Flatten(pool);
    network.push_back(flatten);
    printf("flatten out size %d x %d\n", flatten->out_dim_size(0),
                                         flatten->out_dim_size(1));

    int C = 10; // number of classes

    Affine * fc = new Affine(C, flatten);
    network.push_back(fc);
    printf("fc out size %d x %d\n", fc->out_dim_size(0),
                                    fc->out_dim_size(1));

    SoftMax * softm = new SoftMax(fc);
    network.push_back(softm);
    printf("softm out size %d x %d\n", softm->out_dim_size(0),
                                       softm->out_dim_size(1));

    Image<float> scores(C, N);
    Image<int> labels(N);


    softm->back_propagate(Func(labels));

    // Schedule
    conv->forward.compute_root();
    pool->forward.compute_root();
    fc->forward.compute_root();
    softm->forward.compute_root();
    conv->f_param_grads[0].compute_root();
    conv->f_param_grads[1].compute_root();
    conv->f_in_grad.compute_root();
    fc->f_param_grads[0].compute_root();
    fc->f_param_grads[1].compute_root();
    fc->f_in_grad.compute_root();
    pool->f_in_grad.compute_root();

    conv->f_param_grads[0].print_loop_nest();

    // Build
    std::vector<Func> outs;
    outs.push_back(softm->forward);
    outs.push_back(conv->f_param_grads[0]);
    Pipeline p(outs);

    timeval t1, t2;
    gettimeofday(&t1, NULL);
    p.realize({scores, conv->param_grads[0]});
    gettimeofday(&t2, NULL);

    float time = (t2.tv_sec - t1.tv_sec) +
        (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    printf("First JIT time: %f\n", time);


    gettimeofday(&t1, NULL);
    p.realize({scores, conv->param_grads[0]});
    gettimeofday(&t2, NULL);

    time = (t2.tv_sec - t1.tv_sec) +
        (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    printf("Second JIT time: %f\n", time);

    for (Layer* l: network)
        delete l;

    return 0;
}
