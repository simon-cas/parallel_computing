#include <stdio.h>
#include <immintrin.h>


void sgemv_accum16(float *out, float *weights, int rows, int cols, int col_stride, float *x);

int main(){
    int i;

    float * weights = (float*) calloc(512, sizeof(float));
    float * x = (float*) calloc(16, sizeof(float));
    float * out = (float*) calloc(32, sizeof(float));


    for(i=0;i<512;i++){
        weights[i] = i/511.0;
    }

    for(i=0;i<16;i++){
        x[i] = i/15.0;
    }

    sgemv_accum16(out, weights, 32, 16, 32, x);

     for(i=0;i<32;i++){
        printf("%f,", out[i]);
    }

    return 0;
}


void sgemv_accum16(float *out, float *weights, int rows, int cols, int col_stride, float *x)
{
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      float * restrict y;
      __m256 vy0, vy8;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      for (j=0;j<cols;j++)
      {
         __m256 vxj;
         __m256 vw;
         vxj = _mm256_broadcast_ss(&x[j]);

         vw = _mm256_loadu_ps(&weights[j*col_stride + i]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);


         vw = _mm256_loadu_ps(&weights[j*col_stride + i + 8]);
         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
      }
      _mm256_storeu_ps (&y[0], vy0);
      _mm256_storeu_ps (&y[8], vy8);
   }
}



/*gcc -O3 -m64 -Wall -march=native test.c*/
