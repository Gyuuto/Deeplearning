CC = g++
MPICC = mpic++ -DUSE_MPI
CFLAGS = -O3 -std=c++0x

all: approx_cosine mnist_sample mnist_sample_dist

mnist_sample: mnist_sample.cpp
	${CC} ${CFLAGS} -o mnist_sample mnist_sample.cpp

mnist_sample_dist: mnist_sample_dist.cpp
	${MPICC} ${CFLAGS} -o mnist_sample_dist mnist_sample_dist.cpp

approx_cosine: approx_cosine.cpp
	${CC} ${CFLAGS} -o approx_cosine approx_cosine.cpp

clean:
	rm mnist_sample mnist_sample_dist approx_cosine
