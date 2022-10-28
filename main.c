#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>


#define MAX_RAND_VAL 1000

#define DEFAULT 1
#define OMP1 2
#define OMP2 3
#define INTRIN 4


void rand_fill_vec(float* vec, int n)
{
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        vec[i] = 0.123 + rand() % MAX_RAND_VAL;
    }
}


void vec_sum(float *a, float *b, float *c, int n)
{
	for (int i = 0; i < n; i++)
		c[i] = a[i] + b[i];
}


void vec_sum_omp(float *a, float *b, float *c, int n)
{
    #pragma omp simd
	for (int i = 0; i < n; i++)
		c[i] = a[i] + b[i];
}

void vec_sum_omp2(float *a, float *b, float *c, int n)
{
    #pragma omp parallel for simd
	for (int i = 0; i < n; i++)
		c[i] = a[i] + b[i];
}


void vec_sum_intrin(float *a, float *b, float *c, int n)
{
	for (int i = 0; i < n / 8; i++) {
        __m256 vec1 = _mm256_loadu_ps(a + i * 8);
        __m256 vec2 = _mm256_loadu_ps(b + i * 8);
        __m256 res = _mm256_add_ps(vec1, vec2);
        _mm256_storeu_ps(c + i * 8, res);
    }
}


double vec_sum_test(int n, int cnt, int opt)
{
    double t = 0.0;
    float *vec1 = (float*)malloc(sizeof(float) * n);
    float *vec2 = (float*)malloc(sizeof(float) * n);
    float *res = (float*)malloc(sizeof(float) * n);

    for (int i = 0; i < cnt; i++) {
        rand_fill_vec(vec1, n);
        rand_fill_vec(vec2, n);
        double t1 = omp_get_wtime();
        switch (opt) {
            case DEFAULT:
            vec_sum(vec1, vec2, res, n);
            break;
            case OMP1:
            vec_sum_omp(vec1, vec2, res, n);
            break;
            case OMP2:
            vec_sum_omp2(vec1, vec2, res, n);
            break;
            case INTRIN:
            vec_sum_intrin(vec1, vec2, res, n);
            break;
        }
        double t2 = omp_get_wtime();
        t += (t2 - t1);
    }
    free(vec1);
    free(vec2);
    free(res);
    return t / cnt;
}


void vec_sum_bench(int start, int steps, int opt)
{
    int n = start;
    switch (opt) {
        case DEFAULT:
        printf("testing default\n");
        break;
        case OMP1:
        printf("testing omp simd\n");
        break;
        case OMP2:
        printf("testing omp simd + parallel\n");
        break;
        case INTRIN:
        printf("testing intrinsic\n");
        break;
    }
    for (int i = 0; i < steps; i++) {
        double t = vec_sum_test(n, 5, opt);
        printf("average time (us): %.2lf, data size: %d\n", t * 1000000, n);
        n *= 2;
    }
}


int main(int argc, char *argv[])
{
	if (__builtin_cpu_supports("avx2"))
		printf("supports avx2\n");
	else
		printf("does not support avx2\n");

	if (__builtin_cpu_supports("avx512f"))
		printf("supports avx512\n");
	else
		printf("does not support avx512\n");

    vec_sum_bench(1000, 4, DEFAULT);
    vec_sum_bench(1000, 4, OMP1);
    vec_sum_bench(1000, 4, INTRIN);
    
	
    return 0;
}