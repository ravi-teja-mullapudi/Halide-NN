#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "common.hpp"
#include "io.hpp"
#include "data.pb.h"

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte

cv::Mat ReadImageToCVMat(const string& filename,
        const int height, const int width, const bool is_color) {
    cv::Mat cv_img;
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
            CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
    if (!cv_img_origin.data) {
        LOG(ERROR) << "Could not open or find file " << filename;
        return cv_img_origin;
    }
    if (height > 0 && width > 0) {
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
    } else {
        cv_img = cv_img_origin;
    }
    return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
        const int height, const int width) {
    return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
        const bool is_color) {
    return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
    return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
        std::string en) {
    size_t p = fn.rfind('.');
    std::string ext = p != fn.npos ? fn.substr(p) : fn;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    std::transform(en.begin(), en.end(), en.begin(), ::tolower);
    if ( ext == en )
        return true;
    if ( en == "jpg" && ext == "jpeg" )
        return true;
    return false;
}

bool ReadImageToDatum(const string& filename, const int label,
        const int height, const int width, const bool is_color,
        const std::string & encoding, Datum* datum) {
    cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
    if (cv_img.data) {
        if (encoding.size()) {
            if ( (cv_img.channels() == 3) == is_color && !height && !width &&
                    matchExt(filename, encoding) )
                return ReadFileToDatum(filename, label, datum);
            std::vector<uchar> buf;
            cv::imencode("."+encoding, cv_img, buf);
            datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                        buf.size()));
            datum->set_label(label);
            datum->set_encoded(true);
            return true;
        }
        CVMatToDatum(cv_img, datum);
        datum->set_label(label);
        return true;
    } else {
        return false;
    }
}

bool ReadFileToDatum(const string& filename, const int label,
        Datum* datum) {
    std::streampos size;

    fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
    if (file.is_open()) {
        size = file.tellg();
        std::string buffer(size, ' ');
        file.seekg(0, ios::beg);
        file.read(&buffer[0], size);
        file.close();
        datum->set_data(buffer);
        datum->set_label(label);
        datum->set_encoded(true);
        return true;
    } else {
        return false;
    }
}

cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
    cv::Mat cv_img;
    CHECK(datum.encoded()) << "Datum not encoded";
    const string& data = datum.data();
    std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
    cv_img = cv::imdecode(vec_data, -1);
    if (!cv_img.data) {
        LOG(ERROR) << "Could not decode datum ";
    }
    return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
    cv::Mat cv_img;
    CHECK(datum.encoded()) << "Datum not encoded";
    const string& data = datum.data();
    std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
            CV_LOAD_IMAGE_GRAYSCALE);
    cv_img = cv::imdecode(vec_data, cv_read_flag);
    if (!cv_img.data) {
        LOG(ERROR) << "Could not decode datum ";
    }
    return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
    if (datum->encoded()) {
        cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
        CVMatToDatum(cv_img, datum);
        return true;
    } else {
        return false;
    }
}
bool DecodeDatum(Datum* datum, bool is_color) {
    if (datum->encoded()) {
        cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
        CVMatToDatum(cv_img, datum);
        return true;
    } else {
        return false;
    }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
    CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
    datum->set_channels(cv_img.channels());
    datum->set_height(cv_img.rows);
    datum->set_width(cv_img.cols);
    datum->clear_data();
    datum->clear_float_data();
    datum->set_encoded(false);
    int datum_channels = datum->channels();
    int datum_height = datum->height();
    int datum_width = datum->width();
    int datum_size = datum_channels * datum_height * datum_width;
    std::string buffer(datum_size, ' ');
    for (int h = 0; h < datum_height; ++h) {
        const uchar* ptr = cv_img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < datum_width; ++w) {
            for (int c = 0; c < datum_channels; ++c) {
                int datum_index = (c * datum_height + h) * datum_width + w;
                buffer[datum_index] = static_cast<char>(ptr[img_index++]);
            }
        }
    }
    datum->set_data(buffer);
}

cv::Mat DatumToCVMat(const Datum& datum) {

    if (datum.encoded()) {
        cv::Mat cv_img;
        cv_img = DecodeDatumToCVMatNative(datum);
        return cv_img;
    }

    const string& data = datum.data();

    int datum_channels = datum.channels();
    int datum_height = datum.height();
    int datum_width = datum.width();

    CHECK(datum_channels==3);
   
    cv::Mat cv_img(datum_height, datum_width, CV_8UC3);

    for (int h = 0; h < datum_height; ++h) {
        for (int w = 0; w < datum_width; ++w) {
            for (int c = 0; c < datum_channels; ++c) {
                int datum_index = (c * datum_height + h) * datum_width + w;
                cv_img.at<cv::Vec3b>(h, w)[c] = static_cast<uchar>(data[datum_index]);
            }
        }
    }

    return cv_img;
}

