#!/bin/bash

gcc  lloops2.c cpuidc.c -fopenmp -DTEST_SERIAL -g  -Wconversion -march=native -ffloat-store -lm -lrt -O0   -o lloops

#-march=native