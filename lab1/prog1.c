#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <memory.h>

const long double TAU = 0.001;
const long double E = 1e-5;
const long double PI = 3.14159265358979323846;
const char TRUE = 1;
const char FALSE = 0;
const int NUMBER_OF_ITERATIONS = 5;

void mult_matrix_by_vector(long double* matrix, long double* vector, long double* mult, int N){
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++)
            mult[i] += matrix[j + i * N] * vector[j];
    }
}

char is_correct(long double* current_vector, long double* right_vector, int N){
    char res = TRUE;
    for (int i = 0; i < N; i++)
        if (fabsl(current_vector[i] - right_vector[i]) > E)
            res = FALSE;
    return res;
}

void print_vector(long double* vector, int N){
    for (int i = 0; i < N; i++)
        printf("%f\n", (float)vector[i]);
}

void fill_matrix(long double* matrix, int N)
{
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < N*N; i++){
        if (i % N == i / N)
            matrix[i] = 2;
        else
            matrix[i] = 1;
    }
}

void fill_vector_u(long double* vector_u, int N){
#pragma omp parallel for schedule(static, 1)
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
#pragma omp parallel for schedule(static, 1)
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
#pragma omp parallel for schedule(static, 1)
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
#pragma omp parallel for schedule(static, 1)
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
        exit(1);
    }
    free(new_vector);
}



int main(int argc, char* argv[]){
    struct timeval tv_start,tv_end;
    struct timezone tz;
    if (argc < 2){
        printf("Lack of arguments");
        return 1;
    }
    char *endptr;

    int N = 750;
    int num_of_threads = strtol(argv[1], &endptr, 10);
    if (num_of_threads > 24 || num_of_threads < 1){
        printf("Invalid number of threads\n");
        return 0;
    }
    omp_set_num_threads(num_of_threads);

    long double* matrix_A = (long double*)malloc(N * N * sizeof(long double));
    long double* vector_b = (long double*)malloc(N * sizeof(long double));
    long double* vector_x = (long double*)malloc(N * sizeof(long double));

    double time = 0;
    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        memset(vector_x, 0, N * sizeof(long double));
        memset(vector_b, 0, N * sizeof(long double));
        memset(matrix_A, 0, N * N * sizeof(long double));

        gettimeofday(&tv_start, &tz);

        fill_matrix(matrix_A, N);
        fill_vector_b(matrix_A, vector_b, N);
        fill_vector_x(vector_x, N);

        simple_iteration_method(matrix_A, vector_b, vector_x, N);

        gettimeofday(&tv_end, &tz);
        double new_time = (double)(tv_end.tv_sec - tv_start.tv_sec) + (tv_end.tv_usec - tv_start.tv_usec) * 0.000001;
        time = ((time > new_time) || (time == 0)) ? new_time : time;

    }

    printf("Number of threads: %d\n", num_of_threads);
    printf("Time: %lf seconds\n", time );
    printf("the second element from the result vector: %f\n\n", (float)vector_x[1]);
    free(matrix_A);
    free(vector_b);
    free(vector_x);
    return 0;
}
