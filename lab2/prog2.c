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

void mult_matrix_by_vector(long double* matrix_local, long double* vector_local, long double* mult_local,
                           long double* vector_send, long double* vector_recv, int rank, int num_of_proc, int N){

    int remainder = N % num_of_proc;
    int integer = N / num_of_proc;
    int count_of_str = (rank < remainder) ? (integer + 1) : integer;
    int max_local = (remainder == 0) ? integer :  (integer + 1);
    int cur_rank = rank;
    int cur_size_of_block = count_of_str;

    memset(mult_local, 0, count_of_str * sizeof(long double));
    memcpy(vector_send, vector_local, count_of_str * sizeof(long double));

    int left = (rank - 1 + num_of_proc) % num_of_proc;
    int right = (rank + 1) % num_of_proc;

    for (int i = 0; i < num_of_proc; i++){//циклически по всем процессам пройдем слева направо
        for (int str = 0; str < count_of_str; str++) {//по всем строкам нашей мини матрицы
            for (int k = 0; k < cur_size_of_block; k++) {//по всем элементам вектора данного процесса
                int index;
                if (cur_rank <= remainder)
                    index = cur_rank * (integer + 1) + k;
                else
                    index = N - integer * (num_of_proc - cur_rank) + k;
                mult_local[str] += matrix_local[index + N * str] * vector_send[k];
            }
        }
        if (num_of_proc == 1)
            break;

        MPI_Sendrecv(vector_send, max_local, MPI_LONG_DOUBLE, right, 1, vector_recv, max_local, MPI_LONG_DOUBLE, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cur_rank = (rank - 1 + num_of_proc - i) % num_of_proc;
        cur_size_of_block = (cur_rank < remainder) ? (integer + 1) : integer;
        memcpy(vector_send, vector_recv, cur_size_of_block * sizeof(long double));
    }

    memset(vector_send, 0, max_local * sizeof(long double));
    memset(vector_recv, 0, max_local * sizeof(long double));
}

void print_vector(long double* vector, int N){
    for (int i = 0; i < N; i++)
        printf("%f\n", (float)vector[i]);
}

char is_correct(long double* current_vector, long double* right_vector, int N){
    char res = TRUE;
    for (int i = 0; i < N; i++)
        if (fabsl(current_vector[i] - right_vector[i]) > TAU)
            res = FALSE;
    return res;
}


long double calc_norm(long double* vector, int N)
{
    long double sum = 0;
    for (int i = 0; i < N; i++)
        sum += (long double)(vector[i] * vector[i]);
    return sqrtl(sum);
}


char is_it_the_end(long double* local_matrix, long double* vector_b_local, long double* vector_x_local,
                   long double* vector_send, long double* vector_recv, int rank, int num_of_proc, int N){
    int count_of_str = (rank < (N % num_of_proc)) ? (N / num_of_proc + 1) : (N / num_of_proc);
    long double* new_vector = (long double*)calloc(count_of_str, sizeof(long double));
    mult_matrix_by_vector(local_matrix, vector_x_local, new_vector, vector_send,
                          vector_recv, rank, num_of_proc, N);
    for (int i = 0; i < count_of_str; i++)
        new_vector[i] -= vector_b_local[i];
    char res = FALSE;

    if (rank == 0){
        long double norm1, norm2;
        norm1 = calc_norm(new_vector, count_of_str);
        norm2 = E * calc_norm(vector_b_local, count_of_str);
        if (norm1 < norm2)
            res = TRUE;
    }
    //MPI_Bcast(&norm1, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&norm2, 1, MPI_LONG_DOUBLE, (num_of_proc == 1) ? 0 : 1, MPI_COMM_WORLD);

    MPI_Bcast(&res, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    free(new_vector);
    return res;
}

void simple_iteration_method(long double* local_matrix, long double* vector_b_local, long double* vector_x_local,
                             long double* vector_u_local, long double* vector_send, long double* vector_recv,
                             int rank, int num_of_proc, int N){
    int count_of_str = (rank < (N % num_of_proc)) ? (N / num_of_proc + 1) : (N / num_of_proc);
    long double* new_vector_local = (long double*)calloc(count_of_str, sizeof(long double));
    do{
        mult_matrix_by_vector(local_matrix, vector_x_local, new_vector_local,
                              vector_send, vector_recv, rank, num_of_proc, N);
        for (int i = 0; i < count_of_str; i++){
            new_vector_local[i] = (new_vector_local[i] - vector_b_local[i]) * TAU;
            vector_x_local[i] -= new_vector_local[i];
        }
    }while(!is_it_the_end(local_matrix, vector_b_local, vector_x_local, vector_send, vector_recv, rank, num_of_proc, N));
    if (!is_correct(vector_x_local, vector_u_local, count_of_str)){
        printf("algorithm malfunction!!!\n");
        exit(0);
    }
    free(new_vector_local);
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
    int max_local = (remainder == 0) ? integer :  (integer + 1);
    long double* matrix_A_local = (long double*)malloc(count_of_str * N * sizeof(long double));
    long double* vector_b_local = (long double*)malloc(count_of_str * sizeof(long double));
    long double* vector_x_local = (long double*)malloc(count_of_str * sizeof(long double));
    long double* vector_u_local = (long double*)malloc(count_of_str  * sizeof(long double));

    long double* vector_send = (long double*)calloc(max_local, sizeof(long double));
    long double* vector_recv = (long double*)calloc(max_local, sizeof(long double));
    double time = 0;
    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        double t_start = MPI_Wtime();
        memset(vector_x_local, 0, count_of_str * sizeof(long double));
        memset(vector_b_local, 0, count_of_str * sizeof(long double));
        memset(vector_u_local, 0, count_of_str * sizeof(long double));
        memset(matrix_A_local, 0, count_of_str * N * sizeof(long double));

        for (int j = 0; j < count_of_str; j++){
            int num_of_str = rank + num_of_proc * j;
            vector_u_local[j] = (long double)sinf((float)(2 * PI * num_of_str) / (float)N);
            vector_x_local[j] = 0;
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
            }
        }
        mult_matrix_by_vector(matrix_A_local, vector_u_local, vector_b_local,
                              vector_send, vector_recv, rank, num_of_proc, N);
        simple_iteration_method(matrix_A_local, vector_b_local, vector_x_local, vector_u_local, vector_send,
                                vector_recv, rank, num_of_proc, N);

        double t_end = MPI_Wtime();
        time = ((time > (t_end - t_start)) || (time == 0)) ? (t_end - t_start) : time;
    }
    if (num_of_proc > 1 && rank == 1)
        printf("the second element from the result vector: %f\n", (float)vector_x_local[0]);
    if (num_of_proc == 1)
        printf("the second element from the result vector: %f\n", (float)vector_x_local[1]);
    if (rank == 0)
        printf("Time: %lf seconds\n", time);
    MPI_Finalize();
    free(vector_u_local);
    free(vector_b_local);
    free(vector_x_local);
    free(vector_send);
    free(vector_recv);
    free(matrix_A_local);
    return 0;
}


