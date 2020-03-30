lloops: 
	gcc  lloops2.c cpuidc.c -fopenmp -DPARALLEL -lm -lrt -O3 -march=native  -o lloops

lloops-serial:
	gcc  lloops.c cpuidc.c -lm -lrt -O0 -march=native -o lloops-serial



clean:
	rm lloops LLloops.txt



