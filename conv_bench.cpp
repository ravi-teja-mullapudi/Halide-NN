#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    timeval t1, t2;

    int N = 1; // number of samples/batch_size
    int d_w = 128; // data width
    int d_h = 128; // data height
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
    Func f_in_bound; 
    f_in_bound = BoundaryConditions::constant_exterior(
                                    d_layer->forward, 0,
                                    0, d_w, 0, d_h);
    f_in_bound.compute_root();
    Image<float> W(f_w, f_h, ch, n_f), b(n_f);

    // Define forward
    Func forward;
    Var x, y, z, n;
    RDom r(0, f_w, 0, f_h, 0, ch);
    // Initialize to bias
    forward(x, y, z, n) = b(z);
    forward(x, y, z, n) += W(r.x, r.y, r.z, z) *
                           f_in_bound(x*stride + r.x - pad,
                                      y*stride + r.y - pad,
                                      r.z, n);

    int i_block_size = 8;
    int num_blocks = ch/i_block_size;
    printf("num blocks = %d\n", num_blocks);
    Var i_B, o_B, x_t, y_t, z_t;
    RDom rB(0, f_w, 0, f_h, 0, i_block_size);
    Func f_partial;
    /*
    f_partial(x, y, i_B, n) = 0.0f;
    f_partial(x, y, i_B, n) += W(rB.x, rB.y, (i_B%num_blocks)*i_block_size + rB.z, 
                                  i_B/num_blocks) *
                                  f_in_bound(x*stride + rB.x - pad,
                                             y*stride + rB.y - pad,
                                             (i_B%num_blocks)*i_block_size + rB.z, n);
    */                                             
    f_partial(x, y, i_B, z, n) = 0.0f;
    f_partial(x, y, i_B, z, n) += W(rB.x, rB.y, (i_B)*i_block_size + rB.z, z) *
                                  f_in_bound(x*stride + rB.x - pad,
                                             y*stride + rB.y - pad,
                                             (i_B)*i_block_size + rB.z, n);
    Func f_full;
    RDom rBlocks(0, num_blocks);
    f_full(x, y, z, n) = b(z);
    f_full(x, y, z, n) += f_partial(x, y, rBlocks.x, z, n);

    f_partial.compute_root();
    f_partial.vectorize(x, 8);
    f_partial.update().vectorize(x, 8);
    f_partial.update().split(z, z, z_t, 8);
    f_partial.update().reorder(z_t, i_B, z);
    f_full.compute_root();
    f_full.update().vectorize(x, 8);
    f_full.vectorize(x, 8);
    f_full.print_loop_nest();
    /*
    RDom r(0, 10, 0, 20, 0, 30);
    Var x;
    Func f, g;
    f(x) = 0;
    g(x) = x; 
    f(x) +=  g(r.x) + g(r.y) + g(r.z);

    f.update().reorder(r[2], r[1], r[0]);
    f.print_loop_nest();
    */

    /*
    Var a, b, c, d, e;
    Func f;
    f(a, b, c, d, e) = 0;
    f.print_loop_nest();
    */

    // Schedule
    int schedule = 5;
    switch(schedule) {
        case 1:
            // sequential schedule vectorization
            forward.compute_root();
            //forward.update().vectorize(x, 8);          
            forward.print_loop_nest();
            break;
        case 2:
            // simple multi core parallel schedule
            forward.compute_root();
            forward.update().parallel(z);
            forward.update().vectorize(x, 8);          
            forward.print_loop_nest();
            break;
        case 3:
            // sequential blocking on output filters
            forward.update().split(z, z, z_t, 8);
            forward.update().reorder(z_t, x, y, 
                                          z, n);
            forward.print_loop_nest();
            break;
        case 4:
            // blocking spatially on the outputs
            //forward.update().split(x, x, x_t, 8);
            //forward.update().split(y, y, y_t, 8);
            //forward.update().reorder(x_t, y_t, x, y, 
            //                            z, n);
            //forward.update().unroll(r[0]);
            forward.update().unroll(r[1]);
            forward.update().vectorize(x, 8);          
            forward.print_loop_nest();
            break;
        default:
            printf("meh\n");     
    }
    Image<float> conv_out(conv->out_dim_size(0),
                          conv->out_dim_size(1),
                          conv->out_dim_size(2),
                          conv->out_dim_size(3));
    /*
    Image<float> partial_out(conv->out_dim_size(0),
                             conv->out_dim_size(1),
                             conv->out_dim_size(2) * num_blocks,
                             conv->out_dim_size(3));
    */                         
    std::vector<Func> full_outs;
    full_outs.push_back(f_full);

    std::vector<Func> partial_outs;
    partial_outs.push_back(f_partial);

    Pipeline full(full_outs);
    Pipeline partial(partial_outs);
    for(int it = 0; it < 5; it++) {
        gettimeofday(&t1, NULL);
        full.realize(conv_out);
        gettimeofday(&t2, NULL);

        float time = (t2.tv_sec - t1.tv_sec) +
            (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        printf("time: %f\n", time);
    }
    /*
    for(int it = 0; it < 5; it++) {
        gettimeofday(&t1, NULL);
        partial.realize(partial_out);
        gettimeofday(&t2, NULL);

        float time = (t2.tv_sec - t1.tv_sec) +
            (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        printf("time: %f\n", time);
    }
    */
}
