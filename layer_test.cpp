#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    // Google logging needed for parts of caffe that were extracted

    // Network structure
    // data - conv - reLU - pool - fc - softmax

    // Description of the neural network

    int N = 100; // number of samples/batch_size
    int d_w = 32; // data width
    int d_h = 32; // data height
    int ch = 3; // number of channels

    Image<float> data(32, 32, 3, 100);
    DataLayer d_layer(d_h, d_w, ch, N, data);

    printf("data out size %d x %d x %d x %d\n", d_layer.out_dim_size(0),
                                                d_layer.out_dim_size(1),
                                                d_layer.out_dim_size(2),
                                                d_layer.out_dim_size(3));
    int n_f = 32; // number of filters
    int f_w = 7;  // filter width
    int f_h = 7;  // filter height
    int pad = (f_w-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter evaluated


    Convolutional conv(n_f, f_w, f_h, pad, stride, &d_layer);
    printf("conv out size %d x %d x %d x %d\n", conv.out_dim_size(0),
                                                conv.out_dim_size(1),
                                                conv.out_dim_size(2),
                                                conv.out_dim_size(3));

    ReLU relu(&conv);

    int p_w = 2; // pooling width
    int p_h = 2; // pooling height
    int p_stride = 2; // pooling stride

    MaxPooling pool(p_w, p_h, p_stride, &relu);

    printf("pool out size %d x %d x %d x %d\n", pool.out_dim_size(0),
                                                pool.out_dim_size(1),
                                                pool.out_dim_size(2),
                                                pool.out_dim_size(3));

    Flatten flatten(&pool);

    printf("flatten out size %d x %d\n", flatten.out_dim_size(0),
                                         flatten.out_dim_size(1));

    int C = 10; // number of classes

    Affine fc(C, &flatten);

    printf("fc out size %d x %d\n", fc.out_dim_size(0),
                                    fc.out_dim_size(1));

    SoftMax softm(&fc);

    printf("softm out size %d x %d\n", softm.out_dim_size(0),
                                       softm.out_dim_size(1));

    Image<float> scores(C, N);

    // Schedule
    conv.forward.compute_root();
    pool.forward.compute_root();
    fc.forward.compute_root();
    softm.forward.compute_root();

    // Build
    std::vector<Func> outs;
    outs.push_back(softm.forward);
    Pipeline p(outs);

    timeval t1, t2;
    gettimeofday(&t1, NULL);
    p.realize({scores});
    gettimeofday(&t2, NULL);

    float time = (t2.tv_sec - t1.tv_sec) +
        (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    printf("First JIT time: %f\n", time);


    gettimeofday(&t1, NULL);
    p.realize({scores});
    gettimeofday(&t2, NULL);

    time = (t2.tv_sec - t1.tv_sec) +
        (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    printf("Second JIT time: %f\n", time);

    return 0;
}
