#include "utils.h"
void load_batch(int batch_size, int crop_w, int crop_h,
                Image<float> &data, Image<int> &labels,
                db::Cursor* cur) {

    assert(data.extent(0) == crop_w);
    assert(data.extent(1) == crop_h);
    assert(data.extent(3) == batch_size);
    assert(labels.extent(0) == batch_size);

    for (int n = 0; n < batch_size; n++) {
        Datum datum;
        datum.ParseFromString(cur->value());

        cv::Mat img;
        img = DatumToCVMat(datum);

        assert(crop_w <= img.rows);
        assert(crop_h <= img.cols);

        labels(n) = datum.label();

        for (int h = 0; h < crop_h; h++)
            for (int w = 0; w < crop_w; w++)
                for (int c = 0; c < 3; c++)
                    data(w, h, c, n) = (float)img.at<cv::Vec3b>(h, w)[c];

        cur->Next();
    }
}

void init_gaussian(Image<float> &weights, float mean, float std_dev,
                           std::random_device &rd) {
    //std::mt19937 gen(rd());
    // Fixing the seed for now
    std::mt19937 gen(82387);
    std::normal_distribution<> d(mean, std_dev);

    switch (weights.dimensions()) {
        case 1:
            for (int i = 0; i < weights.extent(0); i++)
                weights(i) = d(gen);
            break;
        case 2:
            for (int i = 0; i < weights.extent(0); i++)
                for (int j = 0; j < weights.extent(1); j++)
                    weights(i, j) = d(gen);
            break;
        case 3:
            for (int i = 0; i < weights.extent(0); i++)
                for (int j = 0; j < weights.extent(1); j++)
                    for (int k = 0; k < weights.extent(2); k++)
                        weights(i, j, k) = d(gen);

            break;
        case 4:
            for (int i = 0; i < weights.extent(0); i++)
                for (int j = 0; j < weights.extent(1); j++)
                    for (int k = 0; k < weights.extent(2); k++)
                        for (int l = 0; l < weights.extent(3); l++)
                            weights(i, j, k, l) = d(gen);
            break;
    }

}

void init_constant(Image<float> &biases, float val) {

    switch (biases.dimensions()) {
        case 1:
            for (int i = 0; i < biases.extent(0); i++)
                biases(i) = val;
            break;
        case 2:
            for (int i = 0; i < biases.extent(0); i++)
                for (int j = 0; j < biases.extent(1); j++)
                    biases(i, j) = val;
            break;
        case 3:
            for (int i = 0; i < biases.extent(0); i++)
                for (int j = 0; j < biases.extent(1); j++)
                    for (int k = 0; k < biases.extent(2); k++)
                        biases(i, j, k) = val;

            break;
        case 4:
            for (int i = 0; i < biases.extent(0); i++)
                for (int j = 0; j < biases.extent(1); j++)
                    for (int k = 0; k < biases.extent(2); k++)
                        for (int l = 0; l < biases.extent(3); l++)
                            biases(i, j, k, l) = val;
            break;
    }
}

void update_with_momentum(Image<float> &param, Image<float> &dparam,
                          Image<float> &update_cache, float momentum,
                          float lr_rate) {
    switch (param.dimensions()) {
        case 1:
            for (int i = 0; i < param.extent(0); i++) {
                update_cache(i) = momentum * update_cache(i) -
                    lr_rate * dparam(i);
                param(i) += update_cache(i);
            }
            break;
        case 2:
            for (int i = 0; i < param.extent(0); i++)
                for (int j = 0; j < param.extent(1); j++) {
                    update_cache(i, j) = momentum * update_cache(i, j) -
                        lr_rate * dparam(i, j);
                    param(i, j) += update_cache(i, j);
                }
            break;
        case 3:
            for (int i = 0; i < param.extent(0); i++)
                for (int j = 0; j < param.extent(1); j++)
                    for (int k = 0; k < param.extent(2); k++) {
                        update_cache(i, j, k) = momentum *
                                                update_cache(i, j, k) -
                                                lr_rate * dparam(i, j, k);
                        param(i, j, k) += update_cache(i, j, k);
                    }
            break;
        case 4:
            for (int i = 0; i < param.extent(0); i++)
                for (int j = 0; j < param.extent(1); j++)
                    for (int k = 0; k < param.extent(2); k++)
                        for (int l = 0; l < param.extent(3); l++) {
                            update_cache(i, j, k, l) = momentum *
                                                       update_cache(i, j, k, l) -
                                                       lr_rate *
                                                       dparam(i, j, k, l);
                            param(i, j, k, l) += update_cache(i, j, k, l);
                        }
            break;
    }
}

void show_filter_weights(int num_f, int f_w, int f_h, int ch, Image<float> &W) {
    // Create window
    cv::namedWindow( "Filter Weights" , cv::WINDOW_NORMAL );
    int sep = 1;
    int total_filters = num_f * ch;
    int grid_size = std::ceil(std::sqrt(total_filters));
    cv::Mat filter_show = cv::Mat::zeros(grid_size * (f_w + sep) -1,
                                         grid_size * (f_h + sep) -1, CV_64FC1);
    for(int n = 0; n < num_f; n++) {
        for(int c = 0; c < ch; c++) {
            int grid_loc_x = (c*num_f + n)/grid_size;
            int grid_loc_y = (c*num_f + n)%grid_size;
            double min_val = W(0, 0, 0, n);
            double max_val = W(0, 0, 0, n);
            for(int i = 0; i < f_h; i++)
                for(int j = 0; j < f_w; j++) {
                    if (W(i, j, c, n) > max_val)
                        max_val = W(i, j, c, n);
                    if (W(i, j, c, n) < min_val)
                        min_val = W(i, j, c, n);
                }
            for(int i = 0; i < f_h; i++)
                for(int j = 0; j < f_w; j++) {
                    double normalized = (W(i, j, c, n) - min_val)/
                                        (max_val - min_val);
                    filter_show.at<double>(grid_loc_x*(f_w + sep) + i,
                                           grid_loc_y*(f_h +sep) + j)
                                              = normalized;
                }
        }
    }
    for(;;) {
        int c;
        c = cv::waitKey(10);
        if( (char)c == 27 )
        { break; }
        imshow( "Filter Weights", filter_show);
    }
}
