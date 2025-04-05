#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

const long double TAU = 0.001;
const long double E = 1e-5;
const long double PI = 3.14159265358979323846;
const char TRUE = 1;
const char FALSE = 0;
const int NUMBER_OF_ITERATIONS = 5;

void mult_matrix_by_vector(long double* matrix_local, long double* vector, long double* mult,
                           int rank, int num_of_proc, int N){
    int remainder = N % num_of_proc;
    int integer = N / num_of_proc;
    int count_of_str = (rank < remainder) ? (integer + 1) : integer;
    for (int i = 0; i < count_of_str; i++){

        int num_of_str = 0;
        if (rank <= remainder)
            num_of_str = rank * (integer + 1) + i;
        else
            num_of_str = N - integer * (num_of_proc - rank) + i;

        for (int j = 0; j < N; j++)
            mult[num_of_str] += matrix_local[j + i * N] * vector[j];
    }
    MPI_Allreduce(MPI_IN_PLACE, mult, N, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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

char is_it_the_end(long double* local_matrix, long double* vector_b, long double* vector_x,
                   int rank, int num_of_proc, int N){
    long double* new_vector = (long double*)calloc(N, sizeof(long double));
    mult_matrix_by_vector(local_matrix, vector_x, new_vector, rank, num_of_proc, N);
    vector_diff(new_vector, vector_b, new_vector, N);
    char res = FALSE;

    if (rank == 0){
        long double norm1, norm2;
        norm1 = calc_norm(new_vector, N);
        norm2 = E * calc_norm(vector_b, N);
        if (norm1 < norm2)
            res = TRUE;
    }
    //MPI_Bcast(&norm1, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&norm2, 1, MPI_LONG_DOUBLE, (num_of_proc == 1) ? 0 : 1, MPI_COMM_WORLD);

    MPI_Bcast(&res, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    free(new_vector);
    return res;
}


void simple_iteration_method(long double* local_matrix, long double* vector_b, long double* vector_x,
                             long double* vector_u, int rank, int num_of_proc, int N){
    long double* new_vector = (long double*)calloc(N, sizeof(long double));
    do{
        mult_matrix_by_vector(local_matrix, vector_x, new_vector, rank, num_of_proc, N);
        for (int i = 0; i < N; i++){
            new_vector[i] = (new_vector[i] - vector_b[i]) * TAU;
            vector_x[i] -= new_vector[i];
        }
    }while(!is_it_the_end(local_matrix, vector_b, vector_x, rank, num_of_proc, N));
    if (!is_correct(vector_x, vector_u, N)){
        printf("algorithm malfunction!!!\n");
        exit(0);
    }
    free(new_vector);
}



int main(int argc, char* argv[]){
    if (argc < 1){
        printf("Lack of arguments");
        return 1;
    }
    char *endptr;
    int N = strtol(argv[1], &endptr, 10);

    MPI_Init(&argc, &argv);
    int num_of_proc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int remainder = N % num_of_proc;
    int integer = N / num_of_proc;
    int count_of_str = (rank < remainder) ? (integer + 1) : integer;
    long double* matrix_A_local = (long double*)malloc(count_of_str * N * sizeof(long double));
    long double* vector_b = (long double*)malloc(N * sizeof(long double));
    long double* vector_x = (long double*)malloc(N * sizeof(long double));
    long double* vector_u = (long double*)malloc(N * sizeof(long double));

    double time = 0;
    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        double t_start = MPI_Wtime();
        memset(vector_x, 0, N * sizeof(long double));
        memset(vector_b, 0, N * sizeof(long double));
        memset(vector_u, 0, N * sizeof(long double));
        memset(matrix_A_local, 0, count_of_str * N * sizeof(long double));

        for (int j = 0; j < N; j++){
            vector_u[j] = (long double)sinf((float)(2 * PI * j) / (float)N);
            vector_x[j] = 0;
        }

        for (int j = 0; j < count_of_str; j++){

            int num_of_str = 0;
            if (rank <= remainder)
                num_of_str = rank * (integer + 1) + j;
            else
                num_of_str = N - integer * (num_of_proc - rank) + j;

            for (int k = 0; k < N; k++){
                if (num_of_str == k)
                    matrix_A_local[k + j * N] = 2;
                else
                    matrix_A_local[k + j * N] = 1;
                vector_b[num_of_str] += matrix_A_local[k + j * N] * vector_u[k];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, vector_b, N, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        simple_iteration_method(matrix_A_local, vector_b, vector_x, vector_u, rank, num_of_proc, N);
        double t_end = MPI_Wtime();
        time = ((time > (t_end - t_start)) || (time == 0)) ? (t_end - t_start) : time;
    }
    if (rank == 0){
        printf("Time: %lf seconds\n", time);
        printf("the second element from the result vector: %f\n", (float)vector_x[1]);
    }
    MPI_Finalize();
    free(vector_u);
    free(vector_b);
    free(vector_x);
    free(matrix_A_local);
    return 0;
}


