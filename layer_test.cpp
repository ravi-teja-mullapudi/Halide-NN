#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    // Google logging needed for parts that were extracted from
    // caffe
    google::InitGoogleLogging(argv[0]);

    // Network structure
    // data - conv - reLU - pool - fc - softmax

    std::vector<Layer*> network;
    float reg = 0.001;

    // Description of the neural network

    int N = 64; // number of samples/batch_size
    int d_w = 32; // data width
    int d_h = 32; // data height
    int ch = 3; // number of channels

    Image<float> data(d_w, d_h, ch, N);
    Image<int> labels(N);


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
                                              stride, reg, d_layer);
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

    Affine * fc = new Affine(C, reg, flatten);
    network.push_back(fc);
    printf("fc out size %d x %d\n", fc->out_dim_size(0),
                                    fc->out_dim_size(1));

    SoftMax * softm = new SoftMax(fc);
    network.push_back(softm);
    printf("softm out size %d x %d\n", softm->out_dim_size(0),
                                       softm->out_dim_size(1));

    Image<float> scores(C, N), loss(1);

    softm->back_propagate(Func(labels));
    Func acc = softm->loss(Func(labels));

    // Schedule
    pool->forward.compute_root().parallel(pool->n);
    conv->forward.compute_root().parallel(conv->n);
    conv->forward.update().parallel(conv->n).vectorize(conv->x, 4);
    fc->forward.compute_root().parallel(fc->n);
    fc->forward.update().parallel(fc->n);
    softm->forward.compute_root();
    conv->f_param_grads[0].compute_root();
    conv->f_param_grads[0].update().parallel(conv->n);//.vectorize(conv->x, 4);
    conv->f_param_grads[1].compute_root();
    conv->f_in_grad.compute_root();
    fc->f_param_grads[0].compute_root();
    fc->f_param_grads[1].compute_root();
    fc->f_in_grad.compute_root().parallel(fc->n);
    fc->f_in_grad.update().parallel(fc->n);
    pool->f_in_grad.compute_root().parallel(pool->n);
    pool->f_in_grad.update().parallel(pool->n);
    acc.compute_root();

    conv->f_param_grads[0].print_loop_nest();

    // Build
    std::vector<Func> train_outs;
    train_outs.push_back(acc);
    train_outs.push_back(softm->forward);
    train_outs.push_back(conv->f_param_grads[0]);
    train_outs.push_back(conv->f_param_grads[1]);
    train_outs.push_back(fc->f_param_grads[0]);
    train_outs.push_back(fc->f_param_grads[1]);
    Pipeline train(train_outs);

    std::vector<Func> test_outs;
    test_outs.push_back(acc);
    test_outs.push_back(softm->forward);
    Pipeline test(test_outs);

    // Loading the cifar db
    string train_path="/home/ravi/Systems/caffe/examples/cifar10/cifar10_train_lmdb";
    string test_path="/home/ravi/Systems/caffe/examples/cifar10/cifar10_test_lmdb";
        
    // Open the lmdb data base
    db::DB* cifar_db_train = db::GetDB("lmdb");
    cifar_db_train->Open(train_path, db::READ);
    db::Cursor* cur_train = cifar_db_train->NewCursor();

    db::DB* cifar_db_test = db::GetDB("lmdb");
    cifar_db_test->Open(test_path, db::READ);
    db::Cursor* cur_test = cifar_db_test->NewCursor();

    std::random_device rd;

    // Training 
    int num_iterartions = 500;
    float lr = 1e-4; 
    float momentum = 0.9;

    // Initialize the biases in the network
    init_constant(conv->params[1], 0.0);
    init_constant(fc->params[1], 0.0);

    // Initialize weights in the network
    init_gaussian(conv->params[0], 0.0001 * 0.0, 0.0001 * 1.0, rd);
    init_gaussian(fc->params[0], 0.0001 * 0.0, 0.0001 * 1.0, rd);

    for (int it = 0; it < num_iterartions; it++) {
        
        // load a batch from the cifar database for training
        load_batch(N, d_w, d_h, data, labels, cur_train);

        timeval t1, t2;
        gettimeofday(&t1, NULL);
        train.realize({loss, scores, conv->param_grads[0], conv->param_grads[1],
                       fc->param_grads[0], fc->param_grads[1]});
        gettimeofday(&t2, NULL);

        float time = (t2.tv_sec - t1.tv_sec) +
            (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        printf("Iteration %d : loss %f, time %f\n", it, loss(0), time);

        for(Layer *l: network) {
            int num_params = l->params.size();
            for (int p = 0; p < num_params; p++) 
                update_with_momentum(l->params[p], l->param_grads[p],
                                     l->params_cache[p], momentum, lr);
        }
    }

    show_filter_weights(conv->num_f, conv->f_w, conv->f_h, 
                        conv->in_ch, conv->params[0]);

    // load a batch for testing
    load_batch(N, d_w, d_h, data, labels, cur_test);

    timeval t1, t2;
    gettimeofday(&t1, NULL);
    test.realize({loss, scores});
    gettimeofday(&t2, NULL);

    float time = (t2.tv_sec - t1.tv_sec) +
        (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    printf("Test: loss %f, time %f\n", loss(0), time);

    int correct = 0;
    for(int sample = 0; sample < N; sample++) {
        int max_label = -1;
        float max_score = 0; 
        for (int c = 0; c < C; c++) {
            if(scores(c, sample) > max_score) {
                max_label = c;
                max_score = scores(c, sample);
            }
        }
        if (max_label == labels(sample)) 
            correct++;
    }

    printf("Accuracy:%f\n", (float)correct/N);        
    
    // Delete all the layers
    for (Layer* l: network)
        delete l;

    cifar_db_train->Close();
    cifar_db_test->Close();
    return 0;
}
