#include "db.hpp"
#include "data.pb.h"
#include "io.hpp"

int main(int argc, char* argv[]) {
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    string path="/home/ravi/data/ilsvrc12_train_lmdb";
    //string path = "/home/ravi/Systems/caffe/examples/cifar10/cifar10_train_lmdb";
    db::DB* cifar_db = db::GetDB("lmdb");
    cifar_db->Open(path, db::READ);

    db::Cursor* cur = cifar_db->NewCursor();
   
    while(true) { 
        Datum datum;
        datum.ParseFromString(cur->value());

        cv::Mat img;
        img = DatumToCVMat(datum);

        // Create window
        cv::namedWindow( "Sample Image" , cv::WINDOW_NORMAL );
        for(;;) {
            int c;
            c = cv::waitKey(10);
            if( (char)c == 27 )
            { break; }
            imshow( "Sample Image", img);
        }
        cur->Next();
    }    
    cifar_db->Close();
    return 0;     
}
