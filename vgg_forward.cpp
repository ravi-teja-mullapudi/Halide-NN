#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    // Google logging needed for parts that were extracted from
    // caffe
    google::InitGoogleLogging(argv[0]);


    // Network structure
    // data -> conv1_1 -> relu1_1 -> conv1_2 -> relu1_2 -> pool1 ->
    // conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> pool2 ->
    // conv3_1 -> relu3_1 -> conv3_2 -> relu3_2 -> conv3_3 -> relu3_3 -> pool3 ->
    // conv4_1 -> relu4_1 -> conv4_2 -> relu4_2 -> conv4_3 -> relu4_3 -> pool4 ->
    // conv5_1 -> relu5_1 -> conv5_2 -> relu5_2 -> conv5_3 -> relu5_3 -> pool5 ->
    // fc6-> relu6 -> droupout6-> fc7 -> relu7 -> dropout7 -> fc8 -> loss

    std::vector<Layer*> network;
    float reg = 0.001;

    // Description of the neural network

    int N = 16; // number of samples/batch_size
    int d_w = 224; // data width
    int d_h = 224; // data height
    int ch = 3; // number of channels

    Image<float> data(d_w, d_h, ch, N);
    Image<int> labels(N);


    DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
    network.push_back(d_layer);
    printf("data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
                                                d_layer->out_dim_size(1),
                                                d_layer->out_dim_size(2),
                                                d_layer->out_dim_size(3));
    int n_f_1 = 64; // number of filters
    int f_w = 3;  // filter width
    int f_h = 3;  // filter height
    int pad = (f_w-1)/2; // padding required to handle boundaries
    int stride = 1; // stride at which the filter evaluated

    Convolutional * conv1_1  = new Convolutional(n_f_1, f_w, f_h, pad,
                                              stride, reg, d_layer);
    conv1_1->o_block_size = 32;
    conv1_1->y_block_size = 16;
    network.push_back(conv1_1);
    printf("conv1_1 out size %d x %d x %d x %d\n", conv1_1->out_dim_size(0),
                                                   conv1_1->out_dim_size(1),
                                                   conv1_1->out_dim_size(2),
                                                   conv1_1->out_dim_size(3));

    ReLU * relu1_1 = new ReLU(conv1_1, 1);
    network.push_back(relu1_1);

    Convolutional * conv1_2  = new Convolutional(n_f_1, f_w, f_h, pad,
                                              stride, reg, relu1_1);

    //relu1_1->forward.compute_at(conv1_2->forward, conv1_2->x);
    //relu1_1->forward.vectorize(relu1_1->x, 8);

    conv1_2->o_block_size = 32;
    conv1_2->y_block_size = 32;
    network.push_back(conv1_2);
    printf("conv1_2 out size %d x %d x %d x %d\n", conv1_2->out_dim_size(0),
                                                   conv1_2->out_dim_size(1),
                                                   conv1_2->out_dim_size(2),
                                                   conv1_2->out_dim_size(3));


    ReLU * relu1_2 = new ReLU(conv1_2);
    network.push_back(relu1_2);

    int p_w = 2; // pooling width
    int p_h = 2; // pooling height
    int p_stride = 2; // pooling stride

    MaxPooling * pool1 = new MaxPooling(p_w, p_h, p_stride, relu1_2);
    network.push_back(pool1);
    printf("pool1 out size %d x %d x %d x %d\n", pool1->out_dim_size(0),
                                                 pool1->out_dim_size(1),
                                                 pool1->out_dim_size(2),
                                                 pool1->out_dim_size(3));

    int n_f_2 = 128;
    Convolutional * conv2_1  = new Convolutional(n_f_2, f_w, f_h, pad,
                                              stride, reg, pool1);
    network.push_back(conv2_1);
    printf("conv2_1 out size %d x %d x %d x %d\n", conv2_1->out_dim_size(0),
                                                   conv2_1->out_dim_size(1),
                                                   conv2_1->out_dim_size(2),
                                                   conv2_1->out_dim_size(3));

    ReLU * relu2_1 = new ReLU(conv2_1);
    network.push_back(relu2_1);

    Convolutional * conv2_2  = new Convolutional(n_f_2, f_w, f_h, pad,
                                              stride, reg, relu2_1);
    network.push_back(conv2_2);
    printf("conv2_2 out size %d x %d x %d x %d\n", conv2_2->out_dim_size(0),
                                                   conv2_2->out_dim_size(1),
                                                   conv2_2->out_dim_size(2),
                                                   conv2_2->out_dim_size(3));
    ReLU * relu2_2 = new ReLU(conv2_2);
    network.push_back(relu2_2);

    MaxPooling * pool2 = new MaxPooling(p_w, p_h, p_stride, relu2_2);
    network.push_back(pool2);
    printf("pool2 out size %d x %d x %d x %d\n", pool2->out_dim_size(0),
                                                 pool2->out_dim_size(1),
                                                 pool2->out_dim_size(2),
                                                 pool2->out_dim_size(3));


    int n_f_3 = 256;
    Convolutional * conv3_1  = new Convolutional(n_f_3, f_w, f_h, pad,
                                              stride, reg, pool2);
    network.push_back(conv3_1);
    printf("conv3_1 out size %d x %d x %d x %d\n", conv3_1->out_dim_size(0),
                                                   conv3_1->out_dim_size(1),
                                                   conv3_1->out_dim_size(2),
                                                   conv3_1->out_dim_size(3));

    ReLU * relu3_1 = new ReLU(conv3_1);
    network.push_back(relu3_1);

    Convolutional * conv3_2  = new Convolutional(n_f_3, f_w, f_h, pad,
                                              stride, reg, relu3_1);
    network.push_back(conv3_2);
    printf("conv3_2 out size %d x %d x %d x %d\n", conv3_2->out_dim_size(0),
                                                   conv3_2->out_dim_size(1),
                                                   conv3_2->out_dim_size(2),
                                                   conv3_2->out_dim_size(3));
    ReLU * relu3_2 = new ReLU(conv3_2);
    network.push_back(relu3_2);

    Convolutional * conv3_3  = new Convolutional(n_f_3, f_w, f_h, pad,
                                              stride, reg, relu3_2);
    network.push_back(conv3_3);
    printf("conv3_3 out size %d x %d x %d x %d\n", conv3_3->out_dim_size(0),
                                                   conv3_3->out_dim_size(1),
                                                   conv3_3->out_dim_size(2),
                                                   conv3_3->out_dim_size(3));
    ReLU * relu3_3 = new ReLU(conv3_3);
    network.push_back(relu3_3);

    MaxPooling * pool3 = new MaxPooling(p_w, p_h, p_stride, relu3_3);
    network.push_back(pool3);
    printf("pool3 out size %d x %d x %d x %d\n", pool3->out_dim_size(0),
                                                 pool3->out_dim_size(1),
                                                 pool3->out_dim_size(2),
                                                 pool3->out_dim_size(3));

    int n_f_4 = 512;
    Convolutional * conv4_1  = new Convolutional(n_f_4, f_w, f_h, pad,
                                              stride, reg, pool3);
    network.push_back(conv4_1);
    printf("conv4_1 out size %d x %d x %d x %d\n", conv4_1->out_dim_size(0),
                                                   conv4_1->out_dim_size(1),
                                                   conv4_1->out_dim_size(2),
                                                   conv4_1->out_dim_size(3));

    ReLU * relu4_1 = new ReLU(conv4_1);
    network.push_back(relu4_1);

    Convolutional * conv4_2  = new Convolutional(n_f_4, f_w, f_h, pad,
                                              stride, reg, relu4_1);
    network.push_back(conv4_2);
    printf("conv4_2 out size %d x %d x %d x %d\n", conv4_2->out_dim_size(0),
                                                   conv4_2->out_dim_size(1),
                                                   conv4_2->out_dim_size(2),
                                                   conv4_2->out_dim_size(3));
    ReLU * relu4_2 = new ReLU(conv4_2);
    network.push_back(relu4_2);

    Convolutional * conv4_3  = new Convolutional(n_f_4, f_w, f_h, pad,
                                              stride, reg, relu4_2);
    network.push_back(conv4_3);
    printf("conv4_3 out size %d x %d x %d x %d\n", conv4_3->out_dim_size(0),
                                                   conv4_3->out_dim_size(1),
                                                   conv4_3->out_dim_size(2),
                                                   conv4_3->out_dim_size(3));
    ReLU * relu4_3 = new ReLU(conv4_3);
    network.push_back(relu4_3);

    MaxPooling * pool4 = new MaxPooling(p_w, p_h, p_stride, relu4_3);
    network.push_back(pool4);
    printf("pool4 out size %d x %d x %d x %d\n", pool4->out_dim_size(0),
                                                 pool4->out_dim_size(1),
                                                 pool4->out_dim_size(2),
                                                 pool4->out_dim_size(3));

    int n_f_5 = 512;
    Convolutional * conv5_1  = new Convolutional(n_f_5, f_w, f_h, pad,
                                              stride, reg, pool4);
    network.push_back(conv5_1);
    printf("conv5_1 out size %d x %d x %d x %d\n", conv5_1->out_dim_size(0),
                                                   conv5_1->out_dim_size(1),
                                                   conv5_1->out_dim_size(2),
                                                   conv5_1->out_dim_size(3));

    ReLU * relu5_1 = new ReLU(conv5_1);
    network.push_back(relu5_1);

    Convolutional * conv5_2  = new Convolutional(n_f_5, f_w, f_h, pad,
                                              stride, reg, relu5_1);
    network.push_back(conv5_2);
    printf("conv5_2 out size %d x %d x %d x %d\n", conv5_2->out_dim_size(0),
                                                   conv5_2->out_dim_size(1),
                                                   conv5_2->out_dim_size(2),
                                                   conv5_2->out_dim_size(3));
    ReLU * relu5_2 = new ReLU(conv5_2);
    network.push_back(relu5_2);

    Convolutional * conv5_3  = new Convolutional(n_f_5, f_w, f_h, pad,
                                              stride, reg, relu5_2);
    network.push_back(conv5_3);
    printf("conv5_3 out size %d x %d x %d x %d\n", conv5_3->out_dim_size(0),
                                                   conv5_3->out_dim_size(1),
                                                   conv5_3->out_dim_size(2),
                                                   conv5_3->out_dim_size(3));
    ReLU * relu5_3 = new ReLU(conv5_3);
    network.push_back(relu5_3);

    MaxPooling * pool5 = new MaxPooling(p_w, p_h, p_stride, relu5_3);
    network.push_back(pool5);
    printf("pool5 out size %d x %d x %d x %d\n", pool5->out_dim_size(0),
                                                 pool5->out_dim_size(1),
                                                 pool5->out_dim_size(2),
                                                 pool5->out_dim_size(3));
    Flatten * flatten = new Flatten(pool5);
    network.push_back(flatten);
    printf("flatten out size %d x %d\n", flatten->out_dim_size(0),
                                         flatten->out_dim_size(1));

    int fc6_out_dim = 4096;

    Affine * fc6 = new Affine(fc6_out_dim, reg, flatten);
    network.push_back(fc6);
    printf("fc6 out size %d x %d\n", fc6->out_dim_size(0),
                                     fc6->out_dim_size(1));

    ReLU * relu6 = new ReLU(fc6);
    network.push_back(relu6);

    // TODO Add drop out

    int fc7_out_dim = 4096;

    Affine * fc7 = new Affine(fc7_out_dim, reg, relu6);
    network.push_back(fc7);
    printf("fc7 out size %d x %d\n", fc7->out_dim_size(0),
                                     fc7->out_dim_size(1));

    ReLU * relu7 = new ReLU(fc7);
    network.push_back(relu7);

    // TODO Add drop out

    int C = 1000;
    Affine * fc8 = new Affine(C, reg, relu7);
    network.push_back(fc8);
    printf("fc8 out size %d x %d\n", fc8->out_dim_size(0),
                                     fc8->out_dim_size(1));

    SoftMax * softm = new SoftMax(fc8);
    network.push_back(softm);
    printf("softm out size %d x %d\n", softm->out_dim_size(0),
                                       softm->out_dim_size(1));

    Image<float> scores(C, N), loss(1);

    //softm->back_propagate(Func(labels));
    Func acc = softm->loss(Func(labels));

    // Schedule
    int sched = 2;
    int vec_len = 8;
    switch(sched) {
        case 1:
            acc.compute_root();

            softm->forward.compute_root();

            fc6->forward.compute_root().parallel(fc6->n);
            fc7->forward.compute_root().parallel(fc7->n);
            fc8->forward.compute_root().parallel(fc8->n);
            fc6->forward.compute_root().update().parallel(fc6->n);
            fc7->forward.compute_root().update().parallel(fc7->n);
            fc8->forward.compute_root().update().parallel(fc8->n);

            flatten->forward.compute_root().parallel(flatten->n);

            pool5->forward.compute_root().parallel(pool5->n);
            //pool5->forward.vectorize(pool5->z, vec_len);

            conv5_1->forward.compute_at(pool5->forward, pool5->n);
            conv5_2->forward.compute_at(pool5->forward, pool5->n);
            conv5_3->forward.compute_at(pool5->forward, pool5->n);

            relu5_1->forward.compute_at(pool5->forward,pool5->n);
            relu5_1->forward.vectorize(relu5_1->x, vec_len);
            relu5_2->forward.compute_at(pool5->forward, pool5->n);
            relu5_2->forward.vectorize(relu5_2->x, vec_len);
            relu5_3->forward.compute_at(pool5->forward, pool5->n);
            relu5_3->forward.vectorize(relu5_3->x, vec_len);

            conv5_1->forward.update().vectorize(conv5_1->x, vec_len);
            conv5_2->forward.update().vectorize(conv5_2->x, vec_len);
            conv5_3->forward.update().vectorize(conv5_3->x, vec_len);

            //pool4->forward.compute_root().parallel(pool4.n).vectorize(pool4.z, vec_len);
            pool4->forward.compute_at(pool5->forward, pool5->n);
            pool4->forward.vectorize(pool4->z, vec_len);

            conv4_1->forward.compute_at(pool5->forward, pool5->n);
            conv4_2->forward.compute_at(pool5->forward, pool5->n);
            conv4_3->forward.compute_at(pool5->forward, pool5->n);


            relu4_1->forward.compute_at(pool5->forward, pool5->n);
            relu4_1->forward.vectorize(relu4_1->x, vec_len);
            relu4_2->forward.compute_at(pool5->forward, pool5->n);
            relu4_2->forward.vectorize(relu4_2->x, vec_len);
            relu4_3->forward.compute_at(pool5->forward, pool5->n);
            relu4_3->forward.vectorize(relu4_3->x, vec_len);

            conv4_1->forward.update().vectorize(conv4_1->x, vec_len);
            conv4_2->forward.update().vectorize(conv4_2->x, vec_len);
            conv4_3->forward.update().vectorize(conv4_3->x, vec_len);

            pool3->forward.compute_root().parallel(pool3->n);
            pool3->forward.vectorize(pool3->z, vec_len);
            conv3_1->forward.compute_at(pool3->forward, pool3->n);
            conv3_2->forward.compute_at(pool3->forward, pool3->n);
            conv3_3->forward.compute_at(pool3->forward, pool3->n);

            relu3_1->forward.compute_at(pool3->forward, pool3->n);
            relu3_1->forward.vectorize(relu3_1->x, vec_len);
            relu3_2->forward.compute_at(pool3->forward, pool3->n);
            relu3_2->forward.vectorize(relu3_2->x, vec_len);
            relu3_3->forward.compute_at(pool3->forward, pool3->n);
            relu3_3->forward.vectorize(relu3_3->x, vec_len);

            conv3_1->forward.update().vectorize(conv3_1->x, vec_len);
            conv3_2->forward.update().vectorize(conv3_2->x, vec_len);
            conv3_3->forward.update().vectorize(conv3_3->x, vec_len);

            pool2->forward.print_loop_nest();
            pool2->forward.compute_root().parallel(pool2->n);
            pool2->forward.vectorize(pool2->z, vec_len);

            conv2_1->forward.compute_at(pool2->forward, pool2->n);
            conv2_2->forward.compute_at(pool2->forward, pool2->n);

            relu2_1->forward.compute_at(pool2->forward, pool2->n);
            relu2_1->forward.vectorize(relu2_1->x, vec_len);
            relu2_2->forward.compute_at(pool2->forward, pool2->n);
            relu2_2->forward.vectorize(relu2_2->x, vec_len);

            conv2_1->forward.update().vectorize(conv2_1->x, vec_len);
            conv2_2->forward.update().vectorize(conv2_2->x, vec_len);

            pool1->forward.compute_at(pool2->forward, pool2->n);
            pool1->forward.vectorize(pool1->z, vec_len);

            conv1_1->forward.compute_at(pool2->forward, pool2->n);
            conv1_2->forward.compute_at(pool2->forward, pool2->n);

            relu1_1->forward.compute_at(pool2->forward, pool2->n);
            relu1_1->forward.vectorize(relu1_1->x, vec_len);
            relu1_2->forward.compute_at(pool2->forward, pool2->n);
            relu1_2->forward.vectorize(relu1_2->x, vec_len);

            conv1_1->forward.update().vectorize(conv1_1->x, vec_len);
            conv1_2->forward.update().vectorize(conv1_2->x, vec_len);
            break;
        case 2:
            break;
    }
    // Build
    std::vector<Func> test_outs;
    test_outs.push_back(acc);
    test_outs.push_back(softm->forward);
    Pipeline test(test_outs);

    for(int it = 0; it < 5; it++) {
        timeval t1, t2;
        gettimeofday(&t1, NULL);
        test.realize({loss, scores});
        gettimeofday(&t2, NULL);

        float time = (t2.tv_sec - t1.tv_sec) +
            (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        printf("time:%f\n", time);
    }

    return 0;
}
