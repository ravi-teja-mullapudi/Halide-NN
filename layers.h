#include "Halide.h"
using namespace Halide;
class Layer {
    public:
        Layer(Layer* in) {
            // The first layer in the pipeline does not have an input layer
            if (in) {
                // Get the halide function that computes values
                // of the input layer
                assert(in->forward.defined());

                // Record the input layer
                in_layer = in;
            }
        }
        // Layer that serves as an input to the current layer
        Layer* in_layer;
        // Number of output dimensions
        virtual  int out_dims() = 0;
        // Size of output dimension i, 0 <= i < out_dims()
        virtual  int out_dim_size( int i) = 0;

        // Storage for layer parameters
        std::vector<Image<float> > params;
        std::vector<Image<float> > param_grads;
        std::vector<Image<float> > params_cache;
        // Halide function that computes the output of the layer
        Func forward;
        // Vector of halide functions which compute the gradients
        // with respect to layer parameters
        std::vector<Func> f_param_grads;
        // Halide function which computes gradient with respect
        // to layer input
        Func f_in_grad;
        // Defines the functions which compute gradient of the objective
        // function with respective to parameters and input. Given a function
        // which computes the derivate of the objective with respect to layer
        // outputs. Does this recursively for the input layer.
        virtual void back_propagate(Func dforward) = 0;

};

class SoftMax: public Layer {
    public:
        Var in_dim, n;
        int num_classes, num_samples;
        // Expects one input layer (num_classes x num_samples)
        SoftMax(Layer* in) : Layer(in) {
            assert(in->out_dims() == 2);

            Func in_f = in_layer->forward;

            num_classes = in->out_dim_size(0);
            num_samples = in->out_dim_size(1);

            // Define forward
            Func exp_max, expo, normalizer;
            RDom r(0, num_classes);
            exp_max(n) = maximum(in_f(r.x, num_samples));
            expo(in_dim, n) = exp(in_f(in_dim, n)
                    - exp_max(n));
            normalizer(n) = cast(in_f.output_types()[0], 0);
            normalizer(n) += expo(r.x, n);
            forward(in_dim, n) = expo(in_dim, n)/normalizer(n);

            // Local schedule
            exp_max.compute_at(forward, n);
            expo.compute_at(forward, n);
            normalizer.compute_at(forward, n);
        }

        void back_propagate(Func labels) {
            if (!f_in_grad.defined()) {
                assert(labels.defined());
                assert(forward.defined());

                Expr label = clamp(labels(n), 0, num_classes -1);
                Expr t = (forward(in_dim, n) - 1)/num_samples;
                Expr f = (forward(in_dim, n)/num_samples);
                f_in_grad(in_dim, n) = select(in_dim == label, t, f);
                in_layer->back_propagate(f_in_grad);
            }
        }

        // Returns a halide function that computes softmax loss given
        // the correct labels for each sample
        Func loss(Func labels) {
            // Should loss be a layer?

            // Check if labels is defined
            assert(labels.defined());
            // Check if the dimensions make sense
            assert(labels.dimensions() == 1);
            // TODO Figure out if there is a scalar type
            Var x;
            Func loss_p;
            RDom r(0, num_samples);
            loss_p(x) = cast(forward.output_types()[0], 0);
            // The clamp is necessary. Otherwise, halide will assume that the
            // label can be anything during bounds inference.
            loss_p(0) += -log(forward(clamp(labels(r.x), 0, num_classes - 1),
                        r.x))/num_samples;
            return loss_p;
        }

        int out_dims() { return 2;}

        int out_dim_size( int i) {
            assert(i < 2);
            int size = 0;
            if (i == 0)
                size = num_classes;
            else if (i == 1)
                size = num_samples;
            return size;
        }
};

class Affine: public Layer {
    public:
        Var in_dim, n, unit_dim;
        // num_units is the number of units in the layer
        // num_inputs is the size of each input sample
        int num_units, num_samples, num_inputs;
        Affine(int _num_units, Layer* in) : Layer(in) {

            Func in_f = in_layer->forward;

            // Create parameters
            num_inputs = in->out_dim_size(0);
            num_samples = in->out_dim_size(1);

            Image<float> W(num_inputs, num_units), b(num_units);
            params.push_back(W); params.push_back(b);

            // Define forward
            RDom r(0, num_inputs);
            // Initialize reduction to baises
            forward(unit_dim, n) = b(unit_dim);
            // Dot product
            forward(unit_dim, n) +=
                in_f(r.x, n) * W(r.x, unit_dim);
        }

        void back_propagate(Func dout) {
            assert(dout.defined());

            if (!f_in_grad.defined()) {
                Func dW, db;

                Image<float> W = params[0];
                Image<float> b = params[1];

                RDom r1(0, num_units);
                // initializing to zero
                f_in_grad(in_dim, n) =
                    cast(dout.output_types()[0], 0);
                f_in_grad(in_dim, n) +=
                    dout(r1.x, n) *  W(in_dim, r1.x);

                RDom r2(0, num_samples);
                // initializing to zero
                dW(in_dim, unit_dim) = cast(dout.output_types()[0], 0);
                Func in_f = in_layer->forward;
                dW(in_dim, unit_dim) +=
                    dout(unit_dim, r2.x) * in_f(in_dim, r2.x);

                f_param_grads.push_back(dW);

                // initializing to zero
                db(unit_dim) = cast(dout.output_types()[0], 0);
                db(unit_dim) += dout(unit_dim, r2.x);

                f_param_grads.push_back(db);

                // Create storage for gradients and caching params
                Image<float> W_grad(num_inputs, num_units);
                param_grads.push_back(W_grad);
                Image<float> W_cache(num_inputs, num_units);
                params_cache.push_back(W_cache);

                Image<float> b_grad(num_units);
                param_grads.push_back(b_grad);
                Image<float> b_cache(num_units);
                params_cache.push_back(b_cache);

                in_layer->back_propagate(f_in_grad);
            }
        }

        int out_dims() { return 2;}

        int out_dim_size( int i) {
            assert(i < 2);
            int size = 0;
            if(i==0)
                size = num_units;
            else if(i==1)
                size = num_samples;
            return size;
        }
};

class DropOut: public Layer {
    public:
        Var x, y, z, w;
        // Threshold value between 0-1 representing the probability
        // with which a unit's output will be dropped
        float thresh;
        // Mask containing the drop out coefficients in the forward pass
        Func mask;
        DropOut(float _thresh, Layer* in) : Layer(in) {

            thresh = _thresh;

            Func in_f = in_layer->forward;

            // Define forward
            // See if there is a better way to do this
            Expr scale = 1.0f/(1.0f - thresh);
            switch(in_layer->out_dims()) {
                case 1:
                    mask(x) = select(random_float() > thresh,
                            scale, 0.0f);
                    forward(x) = mask(x) * in_f(x);
                    break;
                case 2:
                    mask(x, y) = select(random_float() > thresh,
                            scale, 0.0f);
                    forward(x, y) = mask(x, y) * in_f(x, y);
                    break;
                case 3:
                    mask(x, y, z) = select(random_float() > thresh,
                            scale, 0.0f);
                    forward(x, y, z) = mask(x, y, z) * in_f(x, y, z);
                    break;
                case 4:
                    mask(x, y, z, w) = select(random_float() > thresh,
                            scale, 0.0f);
                    forward(x, y, z, w) = mask(x, y, z, w) * in_f(x, y, z, w);
                    break;
                default:
                    assert(0);
            }
            // The mask has to be stored at root. It will be incorrect to
            // recompute the mask since the random number generator will
            // generate different values.
            mask.compute_root();
        }

        void back_propagate(Func dout) {
            assert(dout.defined());
            if(!f_in_grad.defined()) {
                switch(in_layer->out_dims()) {
                    case 1:
                        f_in_grad(x) = dout(x) * mask(x);
                        break;
                    case 2:
                        f_in_grad(x, y) = dout(x, y) * mask(x, y);
                        break;
                    case 3:
                        f_in_grad(x, y, z) = dout(x, y, z) * mask(x, y, z);
                        break;
                    case 4:
                        f_in_grad(x, y, z, w) =
                            dout(x, y, z, w) * mask(x, y, z, w);
                        break;
                    default:
                        assert(0);
                }
                in_layer->back_propagate(f_in_grad);
            }
        }

        int out_dims() { return in_layer->out_dims();}

        int out_dim_size( int i) {
            return in_layer->out_dim_size(i);
        }
};

class ReLU: public Layer {
    public:
        Var x, y, z, w;
        ReLU(float _thresh, Layer* in) : Layer(in) {
            Func in_f = in_layer->forward;
            // Define forward
            switch(in_layer->out_dims()) {
                case 1:
                    forward(x) = max(0, in_f(x));
                    break;
                case 2:
                    forward(x, y) = max(0, in_f(x, y));
                    break;
                case 3:
                    forward(x, y, z) = max(0, in_f(x, y, z));
                    break;
                case 4:
                    forward(x, y, z, w) = max(0, in_f(x, y, z, w));
                    break;
                default:
                    assert(0);
            }
        }

        void back_propagate(Func dout) {
            assert(dout.defined());
            if (!f_in_grad.defined()) {
                Func in_f = in_layer->forward;
                switch(in_layer->out_dims()) {
                    case 1:
                        f_in_grad(x) = dout(x) * select( in_f(x) > 0, 1, 0);
                        break;
                    case 2:
                        f_in_grad(x, y) = dout(x, y) *
                            select( in_f(x, y) > 0, 1, 0);
                        break;
                    case 3:
                        f_in_grad(x, y, z) = dout(x, y, z) *
                            select(in_f(x, y, z) > 0, 1, 0);
                        break;
                    case 4:
                        f_in_grad(x, y, z, w) = dout(x, y, z, w) *
                            select(in_f(x, y, z, w) > 0, 1, 0);
                        break;
                    default:
                        assert(0);
                }
                in_layer->back_propagate(f_in_grad);
            }
        }

        int out_dims() { return in_layer->out_dims();}

        int out_dim_size( int i) {
            return in_layer->out_dim_size(i);
        }
};

class Convolutional: public Layer {
    public:
        Var x, y, z, n;
        // number of channels, height and width of the input to the layer
        int num_samples, in_ch, in_h, in_w;
        // number of filters, filter height, filter width, padding and stride
        int num_f, f_h, f_w, pad, stride;
        Func f_in_bound;
        Convolutional(int _num_f, int _f_h, int _f_w, int _pad, int _stride,
                Layer* in) : Layer(in) {

            assert(in_layer->out_dims() == 4);

            num_samples = in_layer->out_dim_size(3);
            in_ch = in_layer->out_dim_size(2);
            in_h = in_layer->out_dim_size(1);
            in_w = in_layer->out_dim_size(0);

            assert( (in_h + 2 * _pad - _f_h) % _stride == 0);
            assert( (in_w + 2 * _pad - _f_w) % _stride == 0);

            num_f = _num_f; f_h = _f_h; f_w = _f_w;
            pad = _pad; stride = _stride;

            // Boundary condition
            f_in_bound = BoundaryConditions::constant_exterior(
                    in_layer->forward, 0,
                    0, in_w,
                    0, in_h);

            // Create parameters
            Image<float> W(f_w, f_h, in_ch, num_f), b(num_f);
            params.push_back(W); params.push_back(b);

            // Define forward
            RDom r(0, f_w, 0, f_h, 0, in_ch);
            // intialize to bias
            forward(x, y, z, n) = b(z);
            forward(x, y, z, n) += W(r.x, r.y, r.z, z) *
                f_in_bound(x*stride + r.x - pad,
                        y*stride + r.y - pad,
                        r.z, n);
            // This creates a padded input and avoids checking boundary
            // conditions while computing the actual convolution
            f_in_bound.compute_at(forward, n);

        }

        void back_propagate(Func dout) {
            assert(dout.defined());
            if (!f_in_grad.defined()) {
                Func dW, db;

                int out_w = this->out_dim_size(0);
                int out_h = this->out_dim_size(1);

                Image<float> W = params[0];
                Image<float> b = params[1];

                RDom r1(0, out_w, 0, out_h, 0, num_samples);

                // intialize to zero
                dW(x, y, z, n) = cast(dout.output_types()[0], 0);
                dW(x, y, z, n) += dout(r1.x, r1.y, n, r1.z) *
                                       f_in_bound(r1.x*stride + x - pad,
                                                  r1.y*stride + y - pad,
                                                  z, r1.z);

                f_param_grads.push_back(dW);

                // intialize to zero
                db(x) = cast(dout.output_types()[0], 0);
                db(x) += dout(r1.x, r1.y, x, r1.z);

                f_param_grads.push_back(db);

                RDom r2(0, num_f);
                // intialize to zero
                f_in_grad(x, y, z, n) = cast(dout.output_types()[0], 0);
                f_in_grad(x, y, z, n) += dout(x, y, r2.x, n) * W(x, y, z, r2.x);

                // Create storage for gradients and caching params
                Image<float> W_grad(f_w, f_h, in_ch, num_f);
                param_grads.push_back(W_grad);
                Image<float> W_cache(f_w, f_h, in_ch, num_f);
                params_cache.push_back(W_cache);

                Image<float> b_grad(num_f);
                param_grads.push_back(b_grad);
                Image<float> b_cache(num_f);
                params_cache.push_back(b_cache);

                in_layer->back_propagate(f_in_grad);
            }
        }

        int out_dims() { return 4; }

        int out_dim_size( int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0)
                size = (1 + (in_w + 2 * pad - f_w)/stride);
            else if (i == 1)
                size = (1 + (in_h + 2 * pad - f_h)/stride);
            else if (i == 2)
                size = num_f;
            else if (i == 3)
                size = num_samples;
            return size;
        }
};

class MaxPooling: public Layer {
    public:
        // number of color channels in input in_c
        // height and width of the input in_h, in_w
        int num_samples, in_ch, in_h, in_w;
        // height and width of the pool
        // stride at which the pooling is applied
        int p_h, p_w, stride;
        Var x, y, z, n;
        MaxPooling(int _p_w, int _p_h, int _stride, Layer* in) : Layer(in) {
            assert(in_layer->out_dims() == 4);

            num_samples = in_layer->out_dim_size(3);
            in_ch = in_layer->out_dim_size(2);
            in_h = in_layer->out_dim_size(1);
            in_w = in_layer->out_dim_size(0);

            assert((in_h - _p_h) % _stride == 0);
            assert((in_w - _p_w) % _stride == 0);

            p_w = _p_w; p_h = _p_h; stride = _stride;

            // Define forward

            Func in_f = in_layer->forward;
            RDom r(0, p_w, 0, p_h);
            forward(x, y, z, n) = maximum(in_f(x * stride + r.x,
                                               y * stride + r.y,
                                               z, n));

        }

        void back_propagate(Func dout) {
            assert(dout.defined());
            if (!f_in_grad.defined()) {
                Func in_f = in_layer->forward;
                Func pool_argmax;
                RDom r1(0, p_w, 0, p_h);
                pool_argmax(x, y, z, n) = argmax(in_f(x * stride + r1.x,
                                                      y * stride + r1.y,
                                                      z, n));

                pool_argmax.compute_root();
                RDom r2(0, this->out_dim_size(0), 0, this->out_dim_size(1));
                f_in_grad(x, y, z, n) = cast(dout.output_types()[0], 0);

                Expr x_bin = clamp(r2.x * stride +
                                   pool_argmax(r2.x, r2.y, z, n)[0], 0, in_w);
                Expr y_bin = clamp(r2.y * stride +
                                   pool_argmax(r2.x, r2.y, z, n)[1], 0, in_h);

                f_in_grad(x_bin, y_bin, z, n) += dout(r2.x, r2.y, z, n);
                in_layer->back_propagate(f_in_grad);
            }
        }

        int out_dims() { return 4; }

        int out_dim_size( int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0)
                size = 1 + ((in_w - p_w)/stride);
            else if (i == 1)
                size = 1 + ((in_h - p_h)/stride);
            else if (i == 2)
                size = in_layer->out_dim_size(2);
            else if (i == 3)
                size = num_samples;
            return size;
        }
};

class DataLayer: public Layer {
    public:
        int in_w, in_h, in_ch, num_samples;
        Var x, y, z, n;
        DataLayer(int _in_w, int _in_h, int _in_ch, int _num_samples,
                  Image<float> &data) : Layer(0) {

                // Define forward
                forward(x, y, z, n) = data(x, y, z, n);
        }
        // Nothing to propagate
        void back_propagate(Func dout) { assert(dout.defined()); return; }

        int out_dims() { return 4; }

        int out_dim_size( int i) {
            assert(i < 4);
            int size = 0;
            if (i == 0)
                size = in_w;
            else if (i == 1)
                size = in_h;
            else if (i == 2)
                size = in_ch;
            else if (i == 3)
                size = num_samples;
            return size;
        }

};

class Flatten: public Layer {
    public:
        Var x, y, z, n;
        int out_width;
        int num_samples;
        Flatten(Layer *in) : Layer(in) {
            assert(in->out_dims() >= 2 && in->out_dims() <= 4);
            num_samples = in_layer->out_dim_size(in_layer->out_dims() - 1);
            // Define forward
            if (in_layer->out_dims() == 2) {
                out_width = in_layer->out_dim_size(0);
                forward(x, n) = in_layer->forward(x, n);
            } else if (in_layer->out_dims() == 3) {
                int w = in_layer->out_dim_size(0);
                int h = in_layer->out_dim_size(1);
                out_width = w * h;
                forward(x, n) = in_layer->forward(x%w, (x/w), n);
            } else if (in_layer->out_dims() == 4) {
                int w = in_layer->out_dim_size(0);
                int h = in_layer->out_dim_size(1);
                int c = in_layer->out_dim_size(2);
                out_width = w * h * c;
                forward(x, n) = in_layer->forward(x%w, (x/w)%h, x/(w*h), n);
            }

        }

        void back_propagate(Func dout) {
            assert(dout.defined());
            if(!f_in_grad.defined()) {
                if(in_layer->out_dims() == 2)
                    f_in_grad(x, n) = dout(x, n);
                else if(in_layer->out_dims() == 3) {
                    int w = in_layer->out_dim_size(0);
                    f_in_grad(x, y, n) = dout(y*w + x, n);
                } else if (in_layer->out_dims() == 4) {
                    int w = in_layer->out_dim_size(0);
                    int h = in_layer->out_dim_size(1);
                    f_in_grad(x, y, z, n) = dout(z*w*h + y*w + x, n);
                }
                in_layer->back_propagate(f_in_grad);
            }
        }

        int out_dims() { return 2; }

        int out_dim_size( int i) {
            assert(i < 2);
            int size = 0;
            if (i == 0)
                size = out_width;
            else if (i == 1)
                size = num_samples;
            return size;
        }
};
