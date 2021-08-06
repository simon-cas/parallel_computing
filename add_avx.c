#include <stdio.h>
#include <immintrin.h>


int main(){

    /*gcc -O3 -m64 -Wall -mavx  test_avx.c*/
    int i;
    float out[8];
    float m1[8] = {1.0,3.0,1.0,1.0,1.0,1.0,1.0,1.0};
    float x1[8] = {1.0,1.0,8.0,1.0,1.0,1.0,1.0,-1.0};

    __m256 x = _mm256_load_ps(&x1[0]);
    __m256 m = _mm256_load_ps(&m1[0]);

    __m256 y = _mm256_mul_ps(x, m);

    _mm256_store_ps(out, y);


    for(i=0;i<8;i++){
        printf("%f,", out[i]);
    }
    return 0;

}
