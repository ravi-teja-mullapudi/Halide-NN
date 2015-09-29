CXX ?= g++
GXX ?= g++

CXXFLAGS += -std=c++11
LDFLAGS += -L/usr/local/lib -lglog -lgflags -lprotobuf -lleveldb -lsnappy \
		   -llmdb -lboost_system -lm -lopencv_core -lopencv_highgui \
		   -lopencv_imgproc -lboost_thread

all: test_io

data.pb.h data.pb.cc: data.proto
	protoc data.proto --cpp_out=./

io.o: io.cpp io.hpp data.pb.h 
	$(CXX) $(CXXFLAGS) -c -Wall io.cpp

db.o: db.cpp db.hpp data.pb.h
	$(CXX) $(CXXFLAGS) -c -Wall db.cpp

test_io: io.o db.o test_io.cpp data.pb.h data.pb.cc
	$(CXX) $(CXXFLAGS) data.pb.cc test_io.cpp io.o db.o -o test_io.out $(LDFLAGS)

clean:
	rm -f io.o db.o test_io.out data.pb.h data.pb.cc
