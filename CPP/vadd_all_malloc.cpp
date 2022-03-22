// MIT License
 
// Copyright (c) 2022 Advanced Micro Devices, Inc.
 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
#include <stdlib.h>
#include <new>

__global__ void vector_add(float *out, float *a, float *b, int n)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = index; i < n; i += stride)
    out[i] = a[i] + b[i];
}

void host_add(float *out, float *a, float *b, int n)
{
  int i = 0;

  for(i = 0; i < n; i++)
    {
      out[i] = a[i] + b[i];
    }
}

/* Expected to return 0 for matching results, 1 otherwise. */
int check(float *out_h, float *out_d, int n)
{
  int result = 0;

  for(int i = 0; i < n; i++)
    if(fabs(out_h[i] - out_d[i]) > 0.00001)
      {
	result = 1;
	break;
      }
  return result;
}

int main(void)
{
  float *a, *b, *c, *c_gold;
  float *d_a, *d_b, *d_c;
  int n_threads = 256;
  int n_blocks = 1000000;
  int n = n_threads * n_blocks;
  float elapsed = 0.0f;
  hipEvent_t start_ev, stop_ev;

  std::chrono::high_resolution_clock::time_point t1, t2;

  a = (float*)calloc(n,sizeof(float));
  b = (float*)calloc(n,sizeof(float));
  c = (float*)calloc(n,sizeof(float));
  c_gold = (float*)calloc(n,sizeof(float));

  hipMalloc(&d_a,n*sizeof(float));
  hipMalloc(&d_b,n*sizeof(float));
  hipMalloc(&d_c,n*sizeof(float));

  for (int i=0; i < n; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
      c[i] = 0.0f;
    }

  host_add(c_gold,a,b,n);

  hipMemcpy(d_a,a,n*sizeof(float),hipMemcpyHostToDevice);
  hipMemcpy(d_b,b,n*sizeof(float),hipMemcpyHostToDevice);

  vector_add<<<n_blocks, n_threads>>>(d_c, d_a, d_b, n);
  hipDeviceSynchronize();

  printf("==== Memory allocated on CPU (not aligned), accessed by GPU ====\n");

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  printf("CHECK: %d\n",check(c_gold,c,n));

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  free(a); free(b); free(c);

  a = (float *)aligned_alloc(64, n*sizeof(float));
  b = (float *)aligned_alloc(64, n*sizeof(float));
  c = (float *)aligned_alloc(64, n*sizeof(float));

  printf("==== Memory allocated on CPU (aligned 64), accessed by GPU ====\n");

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  free(a); free(b); free(c);
  
  a = (float *)aligned_alloc(32, n*sizeof(float));
  b = (float *)aligned_alloc(32, n*sizeof(float));
  c = (float *)aligned_alloc(32, n*sizeof(float));

  printf("==== Memory allocated on CPU (aligned 32), accessed by GPU ====\n");
  
  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  free(a); free(b); free(c);

  a = (float *)aligned_alloc(16, n*sizeof(float));
  b = (float *)aligned_alloc(16, n*sizeof(float));
  c = (float *)aligned_alloc(16, n*sizeof(float));

  printf("==== Memory allocated on CPU (aligned 16), accessed by GPU ====\n");
  
  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  free(a); free(b); free(c);

  printf("==== Memory allocated on GPU, accessed by GPU ====\n");
  
  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(d_c, d_a, d_b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(d_c, d_a, d_b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  hipFree(d_a); hipFree(d_b); hipFree(d_c);


  return 0;
}
