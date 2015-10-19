CXX ?= g++-5
GXX ?= g++-5

HALIDE_PATH = ../Halide
CXXFLAGS += -std=c++11 -g

INCFLAGS += -I$(HALIDE_PATH)/include
LDFLAGS += -L/usr/local/lib -lglog -lgflags -lprotobuf -lleveldb -lsnappy \
           -llmdb -lboost_system -lm -lopencv_core -lopencv_highgui \
           -lopencv_imgproc -lboost_thread-mt -ldl -lpthread -lz \
           -lHalide -L$(HALIDE_PATH)/bin

all: test

utils.o: utils.h utils.cpp
	$(CXX) $(CXXFLAGS) utils.cpp -c -Wall $(INCFLAGS)

test: layer_test.cpp layers.h utils.o dataloaders/io.o dataloaders/db.o
	$(CXX) $(CXXFLAGS) layer_test.cpp dataloaders/data.pb.cc dataloaders/io.o \
           dataloaders/db.o utils.o -o layer_test.out $(LDFLAGS) $(INCFLAGS)

conv_bench: conv_bench.cpp layers.h utils.o dataloaders/io.o dataloaders/db.o
	$(CXX) $(CXXFLAGS) conv_bench.cpp dataloaders/data.pb.cc dataloaders/io.o \
           dataloaders/db.o utils.o -o conv_bench.out $(LDFLAGS) $(INCFLAGS)
clean:
	rm -f layer_test.out utils.o conv_bench.out
