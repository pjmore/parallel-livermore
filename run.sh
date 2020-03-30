#/bin/bash
P=$1
if [[ -z "$1" ]]; then
P=4
fi
OMP_NUM_THREADS=$P ./lloops