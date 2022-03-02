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

__global__ void atomicAdd_shmem(double *a, double *out, int n)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  __shared__ double mem[64];

  if(index < n)
    {
      if(threadIdx.x < warpSize)
	mem[threadIdx.x] = 0;
      
      __syncthreads();
      
      atomicAdd(&mem[lane],a[index]);

      __syncthreads();

      double val = mem[lane];

      if(wid == 0)
	atomicAdd(out,val);
    }
}

int main(void)
{
  double *a, *out;
  double *d_a, *d_out;
  int n_threads = 256;
  int n_blocks = 20000;
  int n = n_threads * n_blocks;
  double elapsed = 0.0f;
  hipEvent_t start_ev, stop_ev;

  std::chrono::high_resolution_clock::time_point t1, t2;

  hipDevice_t device = -1;
  hipGetDevice(&device);
  
  hipMallocManaged(&a,n*sizeof(double));
  hipMallocManaged(&out,sizeof(double));
  *out = 0.0f;
  //hipMallocManaged(&c,n*sizeof(double));
  hipMalloc(&d_a,n*sizeof(double));
  // hipMalloc(&d_b,n*sizeof(double));
  // hipMalloc(&d_c,n*sizeof(double));

  for (int i=0; i < n; i++)
    a[i] = (double)1.0f;

  hipMemcpy(d_a,a,n*sizeof(double),hipMemcpyHostToDevice);

  hipMemPrefetchAsync( a, n * sizeof(double), device);

  printf("==== Memory allocated on CPU with hipMallocManaged, accessed by GPU ====\n");
    
  t1 = std::chrono::high_resolution_clock::now();

  atomicAdd_shmem<<<n_blocks, n_threads>>>(a, out, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();

  double expected = n;

  if(fabs(*out - expected) > 0.00001)
    {
      printf("Default hipMallocManaged on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,expected);
    }
  else
    {
      printf("Default hipMalloc OK\n");
    }

  double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  hipFree(out);

  hipMallocManaged(&out,sizeof(double));
  hipMemAdvise(out, sizeof(double), hipMemAdviseSetCoarseGrain, 0);

  *out = 0.0f;
  
  atomicAdd_shmem<<<n_blocks, n_threads>>>(a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - expected) > 0.00001)
    {
      printf("CoarseGrain hipMallocManaged+hipMemAdvise on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,expected);
    }
  else
    {
      printf("hipMallocManaged+hipMemAdvise OK\n");
    }
  
  hipFree(out);

  out = (double *) calloc(1,sizeof(double));
  
  atomicAdd_shmem<<<n_blocks, n_threads>>>(a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - expected) > 0.00001)
    {
      printf("malloc default on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,expected);
    }
  else
    {
      printf("malloc default OK\n");
    }

  free(out);
  out = (double *) malloc(sizeof(double));
  hipMemAdvise(out, sizeof(double), hipMemAdviseSetCoarseGrain, 0);
  *out = 0.0f;
  
  atomicAdd_shmem<<<n_blocks, n_threads>>>(a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - expected) > 0.00001)
    {
      printf("malloc+hipMemAdvise on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,expected);
    }
  else
    {
      printf("malloc+hipMemAdvise OK\n");
    }

  n_blocks = 10000;
  n = n_threads * n_blocks;

  // vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  // hipDeviceSynchronize();

  // t2 = std::chrono::high_resolution_clock::now();
  // double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Warm-up elapsed time: %lf sec.\n",times);

  // t1 = std::chrono::high_resolution_clock::now();

  // vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  // hipDeviceSynchronize();

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Elapsed time: %lf sec.\n",times);

  // hipMemcpy(d_a,a,n*sizeof(double),hipMemcpyHostToDevice);
  // hipMemcpy(d_b,b,n*sizeof(double),hipMemcpyHostToDevice);

  // printf("==== Memory allocated on GPU with hipMalloc, accessed by GPU ====\n");

  // t1 = std::chrono::high_resolution_clock::now();

  // vector_add<<<n_blocks, n_threads>>>(d_c, d_a, d_b, n);
  // hipDeviceSynchronize();

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Warm-up elapsed time: %lf sec.\n",times);

  // t1 = std::chrono::high_resolution_clock::now();

  // vector_add<<<n_blocks, n_threads>>>(d_c, d_a, d_b, n);
  // hipDeviceSynchronize();

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Elapsed time: %lf sec.\n",times);

  // printf("==== Memory allocated on GPU with hipMalloc, accessed by CPU ====\n");

  // t1 = std::chrono::high_resolution_clock::now();

  // host_add(d_c, d_a, d_b, n);

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Warm-up elapsed time: %lf sec.\n",times);

  // t1 = std::chrono::high_resolution_clock::now();

  // host_add(d_c, d_a, d_b, n);

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Elapsed time: %lf sec.\n",times);

  // printf("==== Memory allocated on CPU with hipMallocManaged, accessed by CPU (Data on GPU) ====\n");

  // t1 = std::chrono::high_resolution_clock::now();

  // host_add(c, a, b, n);

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Warm-up elapsed time: %lf sec.\n",times);

  // t1 = std::chrono::high_resolution_clock::now();

  // host_add(c, a, b, n);

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Elapsed time: %lf sec.\n",times);

  // hipFree(a);
  // hipFree(b);
  // hipFree(c);

  // hipMallocManaged(&a,n*sizeof(double));
  // hipMallocManaged(&b,n*sizeof(double));
  // hipMallocManaged(&c,n*sizeof(double));

  // for (int i=0; i < n; i++)
  //   {
  //     a[i] = 1.0f;
  //     b[i] = 2.0f;
  //     c[i] = 0.0f;
  //   }

  // printf("==== Memory allocated on CPU with hipMallocManaged, accessed by CPU ====\n");

  // t1 = std::chrono::high_resolution_clock::now();

  // host_add(c, a, b, n);

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Warm-up elapsed time: %lf sec.\n",times);

  // t1 = std::chrono::high_resolution_clock::now();

  // host_add(c, a, b, n);

  // t2 = std::chrono::high_resolution_clock::now();
  // times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  // printf("Elapsed time: %lf sec.\n",times);

  return 0;
}
