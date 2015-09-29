#include "layers.h"
#include<stdio.h>
#include <sys/time.h>

int main(int argc, char **argv) {

    // Description of the neural network
    int N = 100; // number of samples/batch_size

    int N_f = 32; // number of filters
    int f_w = 7;  // filter width
    int f_h = 7;  // filter height
    int pad = (f_w-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter evaluated
    //class Convolutional conv(N_f, f_w, f_h, pad, stride);


    Image<float> I1(5, 5, 1, 1);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            I1(i, j, 0, 0) = i + j;

    class DataLayer data(5, 5, 1, 1, I1);
    data.forward.realize(5, 5, 1, 1);

    Image<float> I2;
    I2 = I1;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            printf("%f ", I2(i, j, 0, 0));
        printf("\n");
    }

    timeval t1, t2;

    Func gradient, blah;
    Var x, y;
    Expr e = x + y;
    gradient(x, y) = e;
    blah(x, y) = e * e;

    // Build
    std::vector<Func> outs;

    outs.push_back(gradient);
    outs.push_back(blah);

    Image<int32_t> out1(800, 600);
    Image<int32_t> out2(80, 60);

    Pipeline p(outs);

    gettimeofday(&t1, NULL);
    p.realize({out1, out2});
    gettimeofday(&t2, NULL);

    float time = (t2.tv_sec - t1.tv_sec) +
        (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    printf("First JIT time: %f\n", time);


    gettimeofday(&t1, NULL);
    p.realize({out1, out2});
    gettimeofday(&t2, NULL);

    time = (t2.tv_sec - t1.tv_sec) +
        (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    printf("Second JIT time: %f\n", time);

    return 0;
}
