#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    timeval t1, t2;

    int N = 100; // number of samples/batch_size
    int d_w = 32; // data width
    int d_h = 32; // data height
    int ch = 3; // number of channels

    Image<float> data(32, 32, 3, 100);
    DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
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
    printf("conv out size %d x %d x %d x %d\n", conv->out_dim_size(0),
                                                conv->out_dim_size(1),
                                                conv->out_dim_size(2),
                                                conv->out_dim_size(3));

    Image<float> conv_out(conv->out_dim_size(0),
                          conv->out_dim_size(1),
                          conv->out_dim_size(2),
                          conv->out_dim_size(3));

    std::vector<Func> outs;
    outs.push_back(conv->forward);

    Pipeline p(outs);

    gettimeofday(&t1, NULL);
    p.realize({conv_out});
    gettimeofday(&t2, NULL);

    float time = (t2.tv_sec - t1.tv_sec) +
        (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    printf("time: %f\n", time);

}
