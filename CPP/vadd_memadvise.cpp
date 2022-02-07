#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>

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

int main(void)
{
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  int n_threads = 256;
  int n_blocks = 1000000;
  int n = n_threads * n_blocks;
  float elapsed = 0.0f;
  hipEvent_t start_ev, stop_ev;

  std::chrono::high_resolution_clock::time_point t1, t2;

  hipMallocManaged(&a,n*sizeof(float));
  hipMallocManaged(&b,n*sizeof(float));
  hipMallocManaged(&c,n*sizeof(float));

  hipMalloc(&d_a,n*sizeof(float));
  hipMalloc(&d_b,n*sizeof(float));
  hipMalloc(&d_c,n*sizeof(float));

  for (int i=0; i < n; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
      c[i] = 0.0f;
    }

  hipDevice_t device = -1;
  hipGetDevice(&device);
  printf("Running on device %d\n",device);
  
  hipMemAdvise( a, n * sizeof(float), hipMemAdviseSetPreferredLocation, device);
  hipMemAdvise( b, n * sizeof(float), hipMemAdviseSetPreferredLocation, device);
  hipMemAdvise( c, n * sizeof(float), hipMemAdviseSetPreferredLocation, device);

  vector_add<<<n_blocks, n_threads>>>(d_c, d_a, d_b, n);
  hipDeviceSynchronize();

  printf("==== Memory allocated on CPU with hipMallocManaged (advise on GPU), accessed by GPU ====\n");

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  t1 = std::chrono::high_resolution_clock::now();

  vector_add<<<n_blocks, n_threads>>>(c, a, b, n);
  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  hipMemcpy(d_a,a,n*sizeof(float),hipMemcpyHostToDevice);
  hipMemcpy(d_b,b,n*sizeof(float),hipMemcpyHostToDevice);

  printf("==== Memory allocated on GPU with hipMalloc, accessed by GPU ====\n");

  hipMemAdvise( a, n * sizeof(float), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
  hipMemAdvise( b, n * sizeof(float), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
  hipMemAdvise( c, n * sizeof(float), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);

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

  printf("==== Memory allocated on GPU with hipMalloc, accessed by CPU ====\n");

  t1 = std::chrono::high_resolution_clock::now();

  host_add(d_c, d_a, d_b, n);

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  t1 = std::chrono::high_resolution_clock::now();

  host_add(d_c, d_a, d_b, n);

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  printf("==== Memory allocated on CPU with hipMallocManaged, accessed by CPU (Data on CPU via hipMemAdvise) ====\n");

  t1 = std::chrono::high_resolution_clock::now();

  host_add(c, a, b, n);

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  t1 = std::chrono::high_resolution_clock::now();

  host_add(c, a, b, n);

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  hipFree(a);
  hipFree(b);
  hipFree(c);

  hipMallocManaged(&a,n*sizeof(float));
  hipMallocManaged(&b,n*sizeof(float));
  hipMallocManaged(&c,n*sizeof(float));

  for (int i=0; i < n; i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
      c[i] = 0.0f;
    }

  printf("==== Memory allocated on CPU with hipMallocManaged, accessed by CPU ====\n");

  t1 = std::chrono::high_resolution_clock::now();

  host_add(c, a, b, n);

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Warm-up elapsed time: %lf sec.\n",times);

  t1 = std::chrono::high_resolution_clock::now();

  host_add(c, a, b, n);

  t2 = std::chrono::high_resolution_clock::now();
  times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("Elapsed time: %lf sec.\n",times);

  return 0;
}
