#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>



unsigned int seed = 1;
long max_numbers = 17;

double zero_one_rand(){
    return (double)rand()/RAND_MAX*2.0-1.0;
}

double* generate_data(){
    srand(1);
    double *in_data = aligned_alloc(4*sizeof(double), max_numbers*sizeof(double));
    for(int i=0; i < max_numbers;i++){
        in_data[i] = zero_one_rand();
    }
    return in_data ;
}

double* generate_constant_data(){
    srand(4);
    double *in_data = aligned_alloc(4*sizeof(double), max_numbers*sizeof(double));
    for(int i=0; i < max_numbers;i++){
        in_data[i] = zero_one_rand();
    }
    return in_data;
}

void serial_kernel(double *data, double* constant_arrays, long n, double sc){
        for ( long k=0 ; k<n-11 ; k++ )
        {
            data[k] = sc + constant_arrays[k]*(sc*constant_arrays[k+10] + sc*constant_arrays[k+11]);
            //x[k] = q + y[k]*( r*z[k+10] + t*z[k+11] );
        }
}

void parallel_kernel(double *data, double* constant_arrays, long n, double sc){
        long k;
        #pragma omp parallel for shared(data, constant_arrays) private(k)  
        for ( k=0 ; k<n-11 ; k++ )
        {
            printf("Operating on %d\n",k);
            data[k] = sc + constant_arrays[k]*(sc*constant_arrays[k+10] + sc*constant_arrays[k+11]);
            //x[k] = q + y[k]*( r*z[k+10] + t*z[k+11] );
        }
}


int main(){
    double *const_data = generate_constant_data();
    double *out_data_serial = generate_data();
    double const_scalar = zero_one_rand();
    double *out_data_parallel = generate_data();
    for(long i = 0; i<max_numbers-11;i++){
        printf("%f   %f\n", out_data_serial[i],out_data_parallel[i]);
    }
    printf("\n\n\n");
    serial_kernel(out_data_serial,const_data, max_numbers, const_scalar);
    parallel_kernel(out_data_parallel,const_data, max_numbers, const_scalar);

    for(long i = 0; i<max_numbers-11;i++){
        printf("%f   %f\n", out_data_serial[i],out_data_parallel[i]);
    }
}

