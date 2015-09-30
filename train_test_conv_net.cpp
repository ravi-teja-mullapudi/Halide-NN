#include <emmintrin.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <random>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "static_image.h"
#include "dataloaders/cifar_loader.h"
#include "conv_net_params.h"
timeval t1, t2;

extern "C" {
#include "halide_conv_net.h"
}

void train(cv::Mat &trainX, cv::Mat &trainY, double reg, double learning_rate,
           double momentum, double learning_rate_decay, int num_epochs);

int main(int argc, char **argv) {

    // Load the data
    cv::Mat trainX, testX;
    cv::Mat trainY, testY;
    trainX = cv::Mat::zeros(1024 * 3, 50000, CV_64FC1);
    testX = cv::Mat::zeros(1024 * 3, 10000, CV_64FC1);
    trainY = cv::Mat::zeros(1, 50000, CV_8UC1);
    testX = cv::Mat::zeros(1, 10000, CV_8UC1);

    read_CIFAR10(trainX, testX, trainY, testY, "./dataloaders");

    // Compute mean of the data
    cv::Mat mean_image = cv::Mat::zeros(1024 * 3, 1, CV_64FC1);
    for(int i = 0; i < trainX.size[0]; i++)
       for(int n = 0; n < trainX.size[1]; n++)
            mean_image.at<double>(i, 0) += trainX.at<double>(i, n)/trainX.size[1];

    // Subtract mean from data
    for(int i = 0; i < trainX.size[0]; i++)
       for(int n = 0; n < trainX.size[1]; n++)
            trainX.at<double>(i, n) -= mean_image.at<double>(i, 0);

#ifdef SHOW_MEAN
    // Create window
    cv::namedWindow( "Mean Image" , cv::WINDOW_NORMAL );
    cv::Mat mean_show = cv::Mat::zeros(32, 32, CV_8UC3);
    for(int c = 0; c < 3; c++)
        for(int i = 0; i < 32; i++)
            for(int j = 0; j < 32; j++)
                // The color channesl seem to be flipped
                mean_show.at<cv::Vec3b>(i, j)[2-c] =
                                (unsigned char) mean_image.at<double>(c*32*32 + i*32 + j, 0);
    for(;;) {
        int c;
        c = cv::waitKey(10);
        if( (char)c == 27 )
        { break; }
        imshow( "Mean Image", mean_show);
    }
#endif

    train(trainX, trainY, 0.001, 1e-4, 0.9, 0.95, 1);
    return 0;
}

void train(cv::Mat &trainX, cv::Mat &trainY, double reg, double learning_rate,
           double momentum, double learning_rate_decay, int num_epochs) {
    // Training using stochaistic gradient descent
    int num_validation = 1000;
    int num_samples = trainX.size[1] - num_validation;
    int iterations_per_epoch = num_samples/N;
    int num_iterartions = num_epochs * iterations_per_epoch;
    int epoch = 0;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> dist_uni(0, num_samples - 1);
    std::normal_distribution<double> dist_norm(0.0, 1.0);

    // Declare inputs of the network. All the constants N, H and C are
    // declared in the two_layer_net_params.h
    // TODO make class for the two layer net and move all network
    // specific code into the class. The trainer can then be independent
    // of the model.
    Image<double> data(32, 32, 3, N);
    Image<unsigned char> labels(N);
    Image<double> conv_W(f_W, f_H, CH, N_f);
    Image<double> conv_W_cache(f_W, f_H, CH, N_f);
    Image<double> conv_b(N_f);
    Image<double> conv_b_cache(N_f);
    Image<double> fc_W(8192, C);
    Image<double> fc_W_cache(8192, C);
    Image<double> fc_b(C);
    Image<double> fc_b_cache(C);

    // Initialize the weights
    for ( int f = 0; f < N_f; f++) {
        conv_b(f) = (double) 0;
        conv_b_cache(f) = (double) 0;
        for ( int c = 0; c < CH; c++)
            for ( int h = 0; h < f_H; h++)
                for ( int w = 0; w < f_W; w++) {
                    conv_W(w, h, c, f) = 0.00001 * dist_norm(generator);
                    conv_W_cache(w, h, c, f) = (double) 0;
                }
    }

    for( int i = 0; i < C; i++) {
        fc_b(i) = (double) 0;
        fc_b_cache(i) = (double) 0;
        for ( int j = 0; j < 8192; j++) {
            fc_W(j, i) = 0.00001 * dist_norm(generator);
            fc_W_cache(j, i) = (double) 0;
        }
    }
    for (int it = 0; it < num_iterartions; it++) {
        std::vector<int> indices;
        // Collect batch size number of sample indices
        // Sampling with replacement
        for (int n = 0; n < N; n++) {
            indices.push_back(dist_uni(generator));
        }
        // Load the data for the corresponding indices into Halide
        // structures
        for(int n = 0; n < N; n++) {
            labels(n) = trainY.at<unsigned char>(0, indices[n]);
            for(int c = 0; c < CH; c++)
                for(int h = 0; h < D_H; h++)
                    for(int w = 0; w < D_W; w++)
                        data(w, h, c, n) = trainX.at<double>(D_W*D_H*c + h * D_W + w, indices[n]);
        }

        // Compute the loss and gradients

        Image<double> probs(C, N), loss(1);
        Image<double> dWconv(f_W, f_H, CH, N_f);
        Image<double> dWfc(8192, C);
        Image<double> dconv_b(N_f);
        Image<double> dfc_b(C);

        gettimeofday(&t1, NULL);
        halide_conv_net(data, labels, conv_W, fc_W, conv_b, fc_b, reg,
                        probs, loss, dWconv, dWfc, dconv_b, dfc_b);
        //halide_conv_net(data, labels, conv_W, fc_W, conv_b, fc_b, reg,
        //                probs, loss);
        gettimeofday(&t2, NULL);

#ifdef SHOW_TIME
        float time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        printf("Iteration %d time: %f\n", it, time);
#endif

        // Update the weights
        for( int i = 0; i < C; i++) {
            // fc_b(i) += -learning_rate * dfc_b(i);
            fc_b_cache(i) = momentum * fc_b_cache(i) - learning_rate * dfc_b(i);
            fc_b(i) += fc_b_cache(i);
            for ( int j = 0; j < 8192; j++) {
                //fc_W(j, i) += -learning_rate * dWfc(j, i);
                fc_W_cache(j, i) = momentum * fc_W_cache(j, i) -
                                    learning_rate * dWfc(j, i);
                fc_W(j, i) += fc_W_cache(j, i);
            }
        }

        for ( int f = 0; f < N_f; f++) {
            conv_b_cache(f) = momentum * conv_b_cache(f) -
                              learning_rate * dconv_b(f);
            conv_b(f) += conv_b_cache(f);
            for ( int c = 0; c < CH; c++)
                for ( int h = 0; h < f_H; h++)
                    for ( int w = 0; w < f_W; w++) {
                        conv_W_cache(w, h, c, f) =
                            momentum * conv_W_cache(w, h, c, f) -
                            learning_rate * dWconv(w, h, c, f);
                        conv_W(w, h, c, f) += conv_W_cache(w, h, c, f);
                    }
        }

        // Periodically check training and validation accuracy
        if (it%100 == 0) {
            int train_correct = 0;
            int val_correct = 0;
            for(int n = 0; n < N; n++){
                int max_prob_label = 0;
                for(int c = 0; c < C; c++){
                    if(probs(c, n) > probs(max_prob_label, n))
                        max_prob_label = c;
                }
                if(max_prob_label == labels(n))
                    train_correct++;

            }
            Image<double> val_data(32, 32, 3, num_validation);
            Image<unsigned char> val_labels(num_validation);
            Image<double> val_probs(C, num_validation);
            for(int n = 0; n < num_validation; n++) {
                val_labels(n) = trainY.at<unsigned char>(0, num_samples + n);
                for(int c = 0; c < CH; c++)
                    for(int h = 0; h < D_H; h++)
                        for(int w = 0; w < D_W; w++)
                            val_data(w, h, c, n) =
                                trainX.at<double>(D_W*D_H*c + h * D_W + w,
                                                  num_samples + n);
            }
            halide_conv_net(val_data, val_labels, conv_W, fc_W, conv_b, fc_b, reg,
                            val_probs, loss, dWconv, dWfc, dconv_b, dfc_b);

            for(int n = 0; n < num_validation; n++){
                int max_prob_label = 0;
                for(int c = 0; c < C; c++){
                    if(val_probs(c, n) > val_probs(max_prob_label, n))
                        max_prob_label = c;
                }
                if(max_prob_label == val_labels(n))
                    val_correct++;

            }
            printf("Validation accuracy: %lf, Training accuracy: %lf, Loss: %lf\n",
                    ((double)val_correct)/num_validation, ((double)train_correct)/N,
                    loss(0));
        }

        bool epoch_end = (it + 1) % iterations_per_epoch == 0;
        if (it > 0 && epoch_end) {
            learning_rate *= learning_rate_decay;
            epoch += 1;
        }

    }
#ifdef SHOW_WEIGHTS
    // Create window
    cv::namedWindow( "Weights" , cv::WINDOW_NORMAL );
    int sep = 1;
    int grid_size = std::ceil(std::sqrt(N_f));
    cv::Mat filter_show = cv::Mat::zeros(grid_size * (f_W + sep) -1,
                                         grid_size * (f_H + sep) -1, CV_64FC3);
    for(int n = 0; n < N_f; n++) {
        int grid_loc_x = n/grid_size;
        int grid_loc_y = n%grid_size;
        double min_val = conv_W(0, 0, 0, n);
        double max_val = conv_W(0, 0, 0, n);
        for(int c = 0; c < CH; c++)
            for(int i = 0; i < f_H; i++)
                for(int j = 0; j < f_W; j++) {
                    if (conv_W(i, j, c, n) > max_val)
                        max_val = conv_W(i, j, c, n);
                    if (conv_W(i, j, c, n) < min_val)
                        min_val = conv_W(i, j, c, n);
                }
        for(int c = 0; c < CH; c++)
            for(int i = 0; i < f_H; i++)
                for(int j = 0; j < f_W; j++) {
                    // The color channesl seem to be flipped
                    double normalized = (conv_W(i, j, c, n) - min_val)/(max_val - min_val);
                    filter_show.at<cv::Vec3d>(grid_loc_x*(f_W + sep) + i,
                                              grid_loc_y*(f_H +sep) + j)[2-c] = normalized;
                }
    }
    for(;;) {
        int c;
        c = cv::waitKey(10);
        if( (char)c == 27 )
        { break; }
        imshow( "Weights", filter_show);
    }
#endif
}
