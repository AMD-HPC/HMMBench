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

__global__ void atomicAdd_worse(int *a, int *out, int n)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index < n)
    {
      atomicAdd(out,a[index]);
    }  
}

void host_add(int *out, int *a, int *b, int n)
{
  int i = 0;

  for(i = 0; i < n; i++)
    {
      out[i] = a[i] + b[i];
    }
}

int main(void)
{
  int *a, *out;
  int *d_a, *d_out;
  int n_threads = 256;
  int n_blocks = 20;
  int n = n_threads * n_blocks;
  int elapsed = 0.0f;
  hipEvent_t start_ev, stop_ev;

  std::chrono::high_resolution_clock::time_point t1, t2;

  hipMallocManaged(&a,n*sizeof(int));
  //hipMallocManaged(&c,n*sizeof(int));
  hipMalloc(&d_a,n*sizeof(int));
  // hipMalloc(&d_b,n*sizeof(int));
  // hipMalloc(&d_c,n*sizeof(int));

  for (int i=0; i < n; i++)
    a[i] = (int)i+1.0f;

  hipMemcpy(d_a,a,n*sizeof(int),hipMemcpyHostToDevice);

  int gauss = n*(n+1)/2.0f;

  /* Default hipHostMalloc */

  hipHostMalloc(&out,sizeof(int),0);
  *out = 0.0f;
    
  atomicAdd_worse<<<n_blocks, n_threads>>>(d_a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - gauss) > 0.00001)
    {
      printf("Default hipHostMalloc default on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,gauss);
    }
  else
    {
      printf("Default hipHostMalloc OK\n");
    }

  hipFree(out);

  /* Portable hipHostMalloc */

  hipHostMalloc(&out,sizeof(int),hipHostMallocPortable);
  *out = 0.0f;
    
  atomicAdd_worse<<<n_blocks, n_threads>>>(d_a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - gauss) > 0.00001)
    {
      printf("Portable hipHostMalloc default on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,gauss);
    }
  else
    {
      printf("Portable hipHostMalloc OK\n");
    }

  hipFree(out);

  /* Coherent hipHostMalloc */

  hipHostMalloc(&out,sizeof(int),hipHostMallocNonCoherent);
  *out = 0.0f;
    
  atomicAdd_worse<<<n_blocks, n_threads>>>(d_a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - gauss) > 0.00001)
    {
      printf("Coherent hipHostMalloc on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,gauss);
    }
  else
    {
      printf("Coherent hipHostMalloc OK\n");
    }

  hipFree(out);

  hipMallocManaged(&out,sizeof(int),hipMemAttachGlobal);
  *out = 0.0f;
  
  atomicAdd_worse<<<n_blocks, n_threads>>>(d_a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - gauss) > 0.00001)
    {
      printf("Default hipMallocManaged on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,gauss);
    }
  else
    {
      printf("Default hipMalloc OK\n");
    }

  hipFree(out);

  hipDevice_t device = -1;
  hipGetDevice(&device);

  hipMallocManaged(&out,sizeof(int),hipMemAttachGlobal);
  hipMemAdvise(out, sizeof(int), hipMemAdviseSetCoarseGrain, device);

  *out = 0.0f;
  
  atomicAdd_worse<<<n_blocks, n_threads>>>(a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - gauss) > 0.00001)
    {
      printf("CoarseGrain hipMallocManaged+hipMemAdvise on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,gauss);
    }
  else
    {
      printf("hipMalloc+hipMemAdvise OK\n");
    }
  
  hipFree(out);

  out = (int *) calloc(1,sizeof(int));
  
  atomicAdd_worse<<<n_blocks, n_threads>>>(a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - gauss) > 0.00001)
    {
      printf("malloc default on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,gauss);
    }
  else
    {
      printf("malloc default OK\n");
    }

  free(out);
  out = (int *) malloc(sizeof(int));
  *out = 0.0f;
  hipMemAdvise(out, sizeof(int), hipMemAdviseSetCoarseGrain, device);
  
  atomicAdd_worse<<<n_blocks, n_threads>>>(a, out, n);
  hipDeviceSynchronize();

  if(fabs(*out - gauss) > 0.00001)
    {
      printf("malloc+hipMemAdvise on fine-grain memory (wrong atomic results)\n");
      printf("Value in out: %f, expected: %f\n",*out,gauss);
    }
  else
    {
      printf("malloc+hipMemAdvise OK\n");
    }

  n_blocks = 10000;
  n = n_threads * n_blocks;

  
  
  // printf("==== Memory allocated on CPU with hipMallocManaged, accessed by GPU ====\n");

  // t1 = std::chrono::high_resolution_clock::now();

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

  // hipMemcpy(d_a,a,n*sizeof(int),hipMemcpyHostToDevice);
  // hipMemcpy(d_b,b,n*sizeof(int),hipMemcpyHostToDevice);

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

  // hipMallocManaged(&a,n*sizeof(int));
  // hipMallocManaged(&b,n*sizeof(int));
  // hipMallocManaged(&c,n*sizeof(int));

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
