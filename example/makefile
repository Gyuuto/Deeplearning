CC = g++
MPICC = mpic++
CFLAGS = -O3 -std=c++0x

all: approx_cosine mnist_sample

mnist_sample: mnist_sample.cpp
	${CC} ${CFLAGS} -o mnist_sample mnist_sample.cpp

approx_cosine: approx_cosine.cpp
	${CC} ${CFLAGS} -o approx_cosine approx_cosine.cpp

clean:
	rm mnist_sample approx_cosine
