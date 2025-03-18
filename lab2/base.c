#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

const long double TAU = 0.001;
const long double E = 1e-5;
const long double PI = 3.14159265358979323846;
const char TRUE = 1;
const char FALSE = 0;
const int NUMBER_OF_ITERATIONS = 5;

void mult_matrix_by_vector(long double* matrix, long double* vector, long double* mult, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++)
            mult[i] += matrix[j + i * N] * vector[j];
    }
}

char is_correct(long double* current_vector, long double* right_vector, int N){
    for (int i = 0; i < N; i++)
        if (fabsl(current_vector[i] - right_vector[i]) > E)
            return FALSE;
    return TRUE;
}

void print_vector(long double* vector, int N){
    for (int i = 0; i < N; i++)
        printf("%f\n", (float)vector[i]);
}

void fill_matrix(long double* matrix, int N)
{
    for (int i = 0; i < N*N; i++){
        if (i % N == i / N)
            matrix[i] = 2;
        else
            matrix[i] = 1;
    }
}

void fill_vector_u(long double* vector_u, int N){
    for (int i = 0; i < N; i++)
        vector_u[i] = (long double)sinf((float)(2 * PI * i) / (float)N);
}

void fill_vector_b(long double* matrix, long double* vector_b, int N)
{
    long double* vector_u = (long double*)calloc(N, sizeof(long double));
    fill_vector_u(vector_u, N);
    mult_matrix_by_vector(matrix, vector_u, vector_b, N);
    free(vector_u);
}

void fill_vector_x(long double* vector_x, int N)
{
    for (int i = 0; i < N; i++)
        vector_x[i] = 0;
}

long double calc_norm(long double* vector, int N)
{
    long double sum = 0;
    for (int i = 0; i < N; i++)
        sum += (long double)(vector[i] * vector[i]);
    return sqrtl(sum);
}

void vector_diff(long double* vector_1, long double* vector_2, long double* diff, int N){
    for(int i = 0; i < N; i++)
        diff[i] = vector_1[i] - vector_2[i];
}

char is_it_the_end(long double* matrix, long double* vector_b, long double* vector_x, int N){//сравнение с E
    long double* new_vector = (long double*)calloc(N, sizeof(long double));
    mult_matrix_by_vector(matrix, vector_x, new_vector, N);
    vector_diff(new_vector, vector_b, new_vector, N);
    char res = FALSE;
    if (calc_norm(new_vector, N) < E * calc_norm(vector_b, N))
        res = TRUE;
    free(new_vector);
    return res;
}

void mult_vector_by_scalar(long double* vector, long double* mult, long double scalar, int N){
    for (int i = 0; i < N; i++)
        mult[i] = vector[i] * scalar;
}

void simple_iteration_method(long double* matrix, long double* vector_b, long double* vector_x, int N){
    long double* new_vector = (long double*)calloc(N, sizeof(long double));
    do{
        mult_matrix_by_vector(matrix, vector_x, new_vector, N);
        vector_diff(new_vector, vector_b, new_vector, N);
        mult_vector_by_scalar(new_vector, new_vector, TAU, N);
        vector_diff(vector_x, new_vector, vector_x, N);
    }while(!is_it_the_end(matrix, vector_b, vector_x, N));
    fill_vector_u(new_vector, N);
    if (!is_correct(vector_x, new_vector, N)){
        printf("algorithm malfunction!!!\n");
        exit(0);
    }
    free(new_vector);
}



int main(int argc, char* argv[]){
    struct timeval tv_start,tv_end;
    struct timezone tz;
    if (argc < 1){
        printf("Lack of arguments");
        return 1;
    }
    char *endptr;
    int N = strtol(argv[1], &endptr, 10);
    long double* matrix_A = (long double*)calloc(N * N, sizeof(long double));
    long double* vector_b = (long double*)calloc(N, sizeof(long double));
    long double* vector_x = (long double*)calloc(N, sizeof(long double));

    double time;
    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        gettimeofday(&tv_start, &tz);

        fill_matrix(matrix_A, N);
        fill_vector_b(matrix_A, vector_b, N);
        fill_vector_x(vector_x, N);

        simple_iteration_method(matrix_A, vector_b, vector_x, N);

        gettimeofday(&tv_end, &tz);
        double new_time = (double)(tv_end.tv_sec - tv_start.tv_sec) + (tv_end.tv_usec - tv_start.tv_usec) * 0.000001;
        time = ((time > new_time) || (time == 0)) ? new_time : time;
    }

    printf("Time: %lf seconds\n", time );
    printf("the second element from the result vector: %f\n", (float)vector_x[1]);
    free(matrix_A);
    free(vector_b);
    free(vector_x);
    return 0;
}

