! MIT License
 
! Copyright (c) 2022 Advanced Micro Devices, Inc.
 
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.

program coherent_vadd
  use iso_c_binding
  use iso_fortran_env, only: real32
  use hipfort
  use hipfort_check
  implicit none

  interface
     subroutine launch(out,a,b,N) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr), value :: a, b, out
       integer, value :: N
     end subroutine
  end interface

  ! type(c_ptr) :: da = c_null_ptr
  ! type(c_ptr) :: db = c_null_ptr
  ! type(c_ptr) :: dout = c_null_ptr

  real(real32),allocatable,target,dimension(:) :: a, b, out

  integer, parameter :: N = 1000000
  integer(c_size_t) :: Nbytes
  integer :: i
  real(real32) :: error
  real(real32), parameter :: error_max = 1.0e-6

  Nbytes = N*c_sizeof(c_float)

  ! Allocate host memory
  allocate(a(N))
  allocate(b(N))
  allocate(out(N))

  ! Initialize host arrays
  a(:) = 1.0
  b(:) = 2.0
  out(:) = 0.0;

  ! call hipCheck(hipMalloc(da,Nbytes))
  ! call hipCheck(hipMalloc(db,Nbytes))
  ! call hipCheck(hipMalloc(dout,Nbytes))

  ! call hipCheck(hipMemcpy(da, c_loc(a), Nbytes, hipMemcpyHostToDevice))
  ! call hipCheck(hipMemcpy(db, c_loc(b), Nbytes, hipMemcpyHostToDevice))

  call launch(c_loc(out),c_loc(a),c_loc(b),N)

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  !call hipCheck(hipMemcpy(c_loc(out), dout, Nbytes, hipMemcpyDeviceToHost))

  ! Verification
  do i = 1,N
     error = abs(out(i) - (a(i)+b(i)) )
     if( error .gt. error_max ) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " Out = ", out(i)
        call exit
     endif
  end do

  ! call hipCheck(hipFree(da))
  ! call hipCheck(hipFree(db))
  ! call hipCheck(hipFree(dout))

  ! Deallocate host memory
  deallocate(a)
  deallocate(b)
  deallocate(out)

  write(*,*) "PASSED!"

end program coherent_vadd
