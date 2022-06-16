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
