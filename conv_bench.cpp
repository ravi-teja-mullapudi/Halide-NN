#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    timeval t1, t2;

    int N = 1; // number of samples/batch_size
    int d_w = 24; // data width
    int d_h = 24; // data height
    int ch = 96; // number of channels

    Image<float> data(d_w, d_h, ch, N);
    DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
    printf("data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
                                                d_layer->out_dim_size(1),
                                                d_layer->out_dim_size(2),
                                                d_layer->out_dim_size(3));
    int n_f = 256; // number of filters
    int f_w = 3;  // filter width
    int f_h = 3;  // filter height
    int pad = (f_w-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter evaluated
    float reg = 0.1;
    Convolutional * conv  = new Convolutional(n_f, f_w, f_h, pad,
                                              stride, reg, d_layer);
    printf("conv out size %d x %d x %d x %d\n", conv->out_dim_size(0),
                                                conv->out_dim_size(1),
                                                conv->out_dim_size(2),
                                                conv->out_dim_size(3));

    /*
    RDom r(0, 10, 0, 20, 0, 30);
    Var x;
    Func f, g;
    f(x) = 0;
    g(x) = x; 
    f(x) +=  g(r.x) + g(r.y) + g(r.z);

    f.update().reorder(r.z, r.y, r.x);
    f.print_loop_nest();
    */
    // Schedule
    int schedule = 1;
    Var x_t, y_t, z_t;
    switch(schedule) {
        case 1:
            // sequential schedule vectorization
            conv->forward.compute_root();
            conv->forward.vectorize(conv->x, 8);          
            conv->forward.update().vectorize(conv->x, 8);          
            conv->forward.print_loop_nest();
            break;
        case 2:
            // simple multi core parallel schedule
            conv->forward.compute_root();
            conv->forward.update().parallel(conv->z);
            conv->forward.update().vectorize(conv->x, 8);          
            conv->forward.print_loop_nest();
            break;
        case 3:
            // sequential blocking on output filters
            conv->forward.update().split(conv->z, conv->z, z_t, 8);
            conv->forward.update().reorder(z_t, conv->x, conv->y, 
                                           conv->z, conv->n);
            conv->forward.print_loop_nest();
            break;
        case 4:
            // blocking spatially on the outputs
            conv->forward.update().split(conv->x, conv->x, x_t, 16);
            conv->forward.update().split(conv->y, conv->y, y_t, 16);
            conv->forward.update().reorder(x_t, y_t, conv->x, conv->y, 
                                           conv->z, conv->n); 
            conv->forward.print_loop_nest();
            break;
        default:
            return 0;     
    }
    Image<float> conv_out(conv->out_dim_size(0),
                          conv->out_dim_size(1),
                          conv->out_dim_size(2),
                          conv->out_dim_size(3));

    std::vector<Func> outs;
    outs.push_back(conv->forward);

    Pipeline p(outs);

    for(int it = 0; it < 5; it++) {
        gettimeofday(&t1, NULL);
        p.realize({conv_out});
        gettimeofday(&t2, NULL);

        float time = (t2.tv_sec - t1.tv_sec) +
            (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        printf("time: %f\n", time);
    }

}
