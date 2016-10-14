CC = g++

WORKINGPATH = $(shell pwd)
SOURCE = $(WORKINGPATH)/source
BUILDPATH = $(WORKINGPATH)/build

#Flags
CFLAGS = -std=c++11

OPENCVINCLUDES = `pkg-config --cflags --libs opencv`

swtSRC = $(SOURCE)/swt.cpp
swtOBJ = $(swtSRC:.c=.o)

all: swt

swt: $(swtOBJ)
	$(CC) $(CFLAGS) -o $(BUILDPATH)/swt $(swtOBJ) $(OPENCVINCLUDES)

$(swtSRC):
	$(CC) $(CFLAGS) -c $(swtSRC) -o swt.o $(OPENCVINCLUDES)
