#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    timeval t1, t2;

    int N = 16; // number of samples/batch_size
    int d_w = 224; // data width
    int d_h = 224; // data height
    int ch = 64; // number of channels

    Image<float> data(d_w, d_h, ch, N);
    DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
    printf("data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
                                                d_layer->out_dim_size(1),
                                                d_layer->out_dim_size(2),
                                                d_layer->out_dim_size(3));

    int n_f = 64; // number of filters
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
    Image<float> W(f_w, f_h, ch, n_f), b(n_f);

    Image<float> conv_out(conv->out_dim_size(0),
                          conv->out_dim_size(1),
                          conv->out_dim_size(2),
                          conv->out_dim_size(3));

    Var x, y, z, n, par;

    // Blocked convolutional layer
    int i_block_size = 16;
    int o_block_size = 32;
    int y_block = 32;
    int num_blocks = ch/i_block_size;
    printf("num blocks = %d\n", num_blocks);
    Var i_B, o_B, x_t, y_t, z_t, z_in_t;
    RDom rB(0, f_w, 0, f_h, 0, i_block_size);
    Func f_partial;
                                                 
    f_partial(x, y, i_B, z, n) = 0.0f;
    f_partial(x, y, i_B, z, n) += W(rB.x, rB.y, (i_B)*i_block_size + rB.z, z) *
                                  f_in_bound(x*stride + rB.x - pad,
                                             y*stride + rB.y - pad,
                                             (i_B)*i_block_size + rB.z, n);
    Func f_blocked;
    RDom rBlocks(0, num_blocks);
    f_blocked(x, y, z, n) = b(z);
    f_blocked(x, y, z, n) += f_partial(x, y, rBlocks.x, z, n);

    // Schedule
    int vec_len = 8;
    int blocked_sched = 1;
    switch(blocked_sched) {
        case 1:
            //f_in_bound.compute_root();
            f_partial.compute_root();
            //f_in_bound.compute_at(f_partial, n);
            //f_partial.compute_at(f_blocked, n);
            f_partial.vectorize(x, vec_len);
            f_partial.fuse(z, n, par).parallel(par);
            //f_partial.parallel(z);
            f_partial.update().reorder(x, y, rB.z);
            f_partial.update().split(y, y, y_t, y_block);
            f_partial.update().vectorize(x, vec_len);
            f_partial.update().split(z, z, z_t, o_block_size);
            f_partial.update().reorder(z_t, i_B, z);
            f_partial.update().reorder(y_t, rB.z, z_t, y, z); 
            f_partial.update().fuse(z, n, par);
            f_partial.update().fuse(i_B, par, par).parallel(par);
            //f_partial.update().parallel(z);
            f_blocked.compute_root();
            f_blocked.fuse(z, n, par).parallel(par);
            //f_blocked.parallel(z);
            f_blocked.update().reorder(x, rBlocks.x);
            f_blocked.update().vectorize(x, vec_len);
            //f_blocked.update().fuse(z, n, par).parallel(par);
            f_blocked.update().parallel(z);
            f_blocked.vectorize(x, vec_len);
            break;
        case  2:
            break;
    }

    f_blocked.print_loop_nest();

    std::vector<Func> full_outs;
    full_outs.push_back(f_blocked);

    std::vector<Func> partial_outs;
    partial_outs.push_back(f_partial);

    Pipeline full(full_outs);
    Pipeline partial(partial_outs);
    /* 
    for(int it = 0; it < 5; it++) {
        gettimeofday(&t1, NULL);
        full.realize(conv_out);
        gettimeofday(&t2, NULL);

        float time = (t2.tv_sec - t1.tv_sec) +
            (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        printf("Blocked time: %f\n", time);
    } 
    */
    // Simple convolution
    Func f_simple;
    RDom r(0, f_w, 0, f_h, 0, ch);
    
    f_simple(x, y, z, n) = b(z);
    
    f_simple(x, y, z, n) += W(r.x, r.y, r.z, z) *
                           f_in_bound(x + r.x - pad,
                                      y + r.y - pad,
                                      r.z, n); 
    /*
    RDom r(0, f_h, 0, ch);
    
    f_simple(x, y, z, n) = b(z);
    
    f_simple(x, y, z, n) += W(0, r.x, r.y, z) *
                           f_in_bound(x + 0 - pad,
                                      y + r.x - pad,
                                      r.y, n) +
                           W(1, r.x, r.y, z) *
                           f_in_bound(x + 1 - pad,
                                   y + r.x - pad,
                                   r.y, n) +
                           W(2, r.x, r.y, z) *
                           f_in_bound(x + 2 - pad,
                                   y + r.x - pad,
                                   r.y, n);                                    
                                   */

    // Schedule
    int schedule = 2;
    switch(schedule) {
        case 1:
            // sequential schedule vectorization
            f_in_bound.compute_root();
            f_simple.update().reorder(x, y, r.z); 
            f_simple.update().vectorize(x, vec_len);          
            f_simple.vectorize(x, vec_len);          
            break;
        case 2:
            // blocking spatially with vectorization
            //f_in_bound.compute_at(f_simple, par);
            f_in_bound.compute_at(f_simple, z_t);
            f_simple.compute_root();
            f_simple.fuse(z, n, par).parallel(par);
            f_simple.update().reorder(x, y, r.z); 
            f_simple.update().split(y, y, y_t, y_block);
            f_simple.update().split(z, z, z_t, o_block_size);
            f_simple.update().reorder(y_t, z_t, y, r.z, z); 
            f_simple.update().vectorize(x, vec_len);          
            f_simple.update().unroll(r.x);          
            f_simple.update().unroll(r.y);          
            f_simple.update().fuse(z, n, par).parallel(par);
            //f_simple.update().fuse(y, par, par).parallel(par);
            //f_simple.update().parallel(z);          
            break;
        case 3:
            // blocking on output filters
            f_in_bound.compute_root();
            f_simple.update().split(z, z, z_t, o_block_size);
            f_simple.update().reorder(x, y, r.z); 
            break;
        case 4:
            // blocking spatially on the outputs
            //f_simple.update().split(x, x, x_t, 8);
            //f_simple.update().split(y, y, y_t, 8);
            //f_simple.update().reorder(x_t, y_t, x, y, 
            //                            z, n);
            //f_simple.update().unroll(r[0]);
            f_in_bound.compute_root();
            f_simple.update().unroll(r[1]);
            f_simple.update().vectorize(x, vec_len);          
            f_simple.update().reorder(x, y, r.z); 
            break;
        default:
            return 0;     
    }
    f_simple.print_loop_nest();

    /*
    Image<float> partial_out(conv->out_dim_size(0),
                             conv->out_dim_size(1),
                             conv->out_dim_size(2) * num_blocks,
                             conv->out_dim_size(3));
    */                         
    std::vector<Func> simple_outs;
    simple_outs.push_back(f_simple);
    for(int it = 0; it < 5; it++) {
        gettimeofday(&t1, NULL);
        f_simple.realize(conv_out);
        gettimeofday(&t2, NULL);

        float time = (t2.tv_sec - t1.tv_sec) +
            (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        printf("Simple time: %f\n", time);
    }
}
