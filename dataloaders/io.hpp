#include "common.hpp"
#include "data.pb.h"

using ::google::protobuf::Message;

bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
      return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
            const int height, const int width, const bool is_color,
                const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
            const int height, const int width, const bool is_color, Datum* datum) {
      return ReadImageToDatum(filename, label, height, width, is_color,
                                        "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
            const int height, const int width, Datum* datum) {
      return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
            const bool is_color, Datum* datum) {
      return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
            Datum* datum) {
      return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
            const std::string & encoding, Datum* datum) {
      return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
            const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
            const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
            const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);
cv::Mat DatumToCVMat(const Datum& datum);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

