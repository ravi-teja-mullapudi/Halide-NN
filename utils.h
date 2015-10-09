#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <random>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "dataloaders/db.hpp"
#include "dataloaders/data.pb.h"
#include "dataloaders/io.hpp"

#include "Halide.h"
using namespace Halide;
void load_batch(int batch_size, int crop_w, int crop_h,
                Image<float> &data, Image<int> &labels,
                db::Cursor* cur);
void init_gaussian(Image<float> &weights, float mean, float std_dev,
                           std::random_device &rd);
void init_constant(Image<float> &biases, float val);
void update_with_momentum(Image<float> &param, Image<float> &dparam,
                          Image<float> &update_cache, float momentum,
                          float lr_rate);
void show_filter_weights(int num_f, int f_w, int f_h, int ch, Image<float> &W);
