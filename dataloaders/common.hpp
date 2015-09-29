#include <string>

#include <boost/shared_ptr.hpp>
#include <glog/logging.h>

#include <unistd.h>
#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
    private:\
  classname(const classname&);\
  classname& operator=(const classname&)

using std::string;
using std::fstream;
using std::ios;
