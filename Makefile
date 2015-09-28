CXX ?= g++
GXX ?= g++

HALIDE_PATH = /home/ravi/Systems/Halide
CXXFLAGS += -std=c++11

LDFLAGS += -L/usr/local/lib -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system \
           -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -ldl -lpthread -lz \
           -lHalide -L$(HALIDE_PATH)/bin -I$(HALIDE_PATH)/include

all: test 

test: layer_test.cpp layers.h
	$(CXX) $(CXXFLAGS) layer_test.cpp -o layer_test.out $(LDFLAGS)

clean:
	rm -f layer_test.out
