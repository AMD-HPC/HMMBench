#include<cstdio>

#pragma omp requires unified_shared_memory

#define N 1024

/// Test of usm mode without memory mapping

int main() {
  int n = N;
  int *a = new int[n];

  for(int i = 0; i < n; i++)
    a[i] = -1;
  
  #pragma omp target teams distribute parallel for
  for(int i = 0; i < n; i++)
    a[i] = i;

  int err = 0;
  for(int i = 0; i < n; i++)
    if (a[i] != i) {
      err++;
      printf("At %d, got %d, expected %d\n", i, a[i], i);
      if (err > 10) return err;
    }

  return err;
}
