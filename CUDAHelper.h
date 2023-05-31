//[]---------------------------------------------------------------[]
//|                                                                 |
//| Copyright (C) 2014, 2019 Orthrus Group.                         |
//|                                                                 |
//| This software is provided 'as-is', without any express or       |
//| implied warranty. In no event will the authors be held liable   |
//| for any damages arising from the use of this software.          |
//|                                                                 |
//| Permission is granted to anyone to use this software for any    |
//| purpose, including commercial applications, and to alter it and |
//| redistribute it freely, subject to the following restrictions:  |
//|                                                                 |
//| 1. The origin of this software must not be misrepresented; you  |
//| must not claim that you wrote the original software. If you use |
//| this software in a product, an acknowledgment in the product    |
//| documentation would be appreciated but is not required.         |
//|                                                                 |
//| 2. Altered source versions must be plainly marked as such, and  |
//| must not be misrepresented as being the original software.      |
//|                                                                 |
//| 3. This notice may not be removed or altered from any source    |
//| distribution.                                                   |
//|                                                                 |
//[]---------------------------------------------------------------[]
//
// OVERVIEW: CUDAHelper.h
// ========
// Classes and functions for CUDA utilities.
//
// Author: Paulo Pagliosa
// Last revision: 15/03/2019

#ifndef __CUDAHelper_h
#define __CUDAHelper_h

#include "SoA.h"
#include <iostream>
#include <stdexcept>
#include "cuda_runtime.h"

namespace cg
{ // begin namespace cg

namespace cuda
{ // begin namespace cuda

void error(const char*, ...);
void checkError(cudaError_t, const char*, int);
void checkLastError(const char*, const char*, int);

#define checkCudaError(err) cuda::checkError(err, __FILE__, __LINE__)
#define checkLastCudaError(msg) cuda::checkLastError(msg, __FILE__, __LINE__)

inline void
reset()
{
  cudaDeviceReset();
}

inline void
initialize(int deviceId = -1)
{
  int count;

  checkCudaError(cudaGetDeviceCount(&count));
  if (count == 0)
    error("No devices supporting CUDA");
  if (deviceId < 0)
    deviceId = 0;
  else if (deviceId > count - 1)
    error("Device %d is not a valid GPU.\n", deviceId);

  cudaDeviceProp deviceProp;

  checkCudaError(cudaGetDeviceProperties(&deviceProp, deviceId));
  if (deviceProp.computeMode == cudaComputeModeProhibited)
    error("Device %d is running in compute mode prohibited", deviceId);
  if (deviceProp.major < 1)
    error("Device %d does not support CUDA", deviceId);
  checkCudaError(cudaSetDevice(deviceId));
  printf("Using CUDA device %d: %s\n", deviceId, deviceProp.name);
}

inline void
synchronize()
{
  checkCudaError(cudaDeviceSynchronize());
}

inline void
allocate(void*& ptr, size_t size)
{
  checkCudaError(cudaMalloc((void**)&ptr, size));
}

inline void
free(void* ptr)
{
  checkCudaError(cudaFree(ptr));
}

template <typename T>
inline void
allocate(T*& ptr, size_t count)
{
  allocate((void*&)ptr, count * sizeof(T));
}

template <typename T>
inline void
free(T*& ptr)
{
  free((void*&)ptr);
}

inline void
copyToHost(void* dst, const void* src, size_t size)
{
  checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

template <typename T>
inline void
copyToHost(T* dst, const T* src, size_t count)
{
  copyToHost((void*)dst, (const void*)src, count * sizeof(T));
}

inline void
copyToDevice(void* dst, const void* src, size_t size)
{
  checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

template <typename T>
inline void
copyToDevice(T* dst, const T* src, size_t count)
{
  copyToDevice((void*)dst, (const void*)src, count * sizeof(T));
}

template <typename T>
inline void
newCopyToDevice(T*& dst, const T* src, size_t n)
{
  allocate<T>(dst, n);
  copyToDevice<T>(dst, src, n);
}

inline void
deviceCopy(void* dst, const void* src, size_t size)
{
  checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

template<typename T>
inline void
deviceCopy(T* dst, const T* src, size_t count)
{
  deviceCopy((void*)dst, (const void*)src, count * sizeof(T));
}

inline void
deviceSet(void* ptr, int value, size_t size)
{
  checkCudaError(cudaMemset(ptr, value, size));
}

inline void
copyToSymbol(const void* dst, const void* src, size_t size)
{
  checkCudaError(cudaMemcpyToSymbol(dst, src, size));
}

template <typename T>
inline void
copyToSymbol(const T* dst, const T* src, size_t count)
{
  copyToSymbol((const void*)dst, (const void*)src, sizeof(T) * count);
}

template <typename T>
inline void
copyToSymbol(const T& dst, const T& src)
{
  copyToSymbol<T>(&dst, &src, 1);
}

template <typename T>
void
dump(const char* s, const T* data, size_t count, std::ostream& os = std::cout)
{
  T* hData = new T[count];

  copyToHost<T>(hData, data, count);
  os << s << ": ";
  for (size_t i = 0; i < count; ++i)
    os << hData[i] << ' ';
  os << '\n';
  delete []hData;
}


//////////////////////////////////////////////////////////
//
// ArrayAllocator: CUDA array allocator class
// ==============
class ArrayAllocator
{
public:
  template <typename T>
  static T* allocate(size_t count)
  {
    T* ptr;

    cuda::allocate<T>(ptr, count);
    return ptr;
  }

  template <typename T>
  static void free(T* ptr)
  {
    cuda::free<T>(ptr);
  }

}; // ArrayAllocator


//////////////////////////////////////////////////////////
//
// Buffer: CUDA buffer class
// ======
template <typename T>
class Buffer
{
public:
  using value_type = T;

  Buffer() = default;

  Buffer(size_t count, const T* hData = nullptr)
  {
    if (count == 0)
      throw std::logic_error("cuda::Buffer ctor: bad size");
    allocate<T>(_data, _count = count);
    if (hData != nullptr)
      copyToDevice<T>(_data, hData, count);
  }

  Buffer(const Buffer<T>&) = delete;
  Buffer<T>& operator =(const Buffer<T>&) = delete;

  Buffer(Buffer<T>&& other):
    _data{other._data}, _count{other._count}
  {
    other._data = nullptr;
    other._count = 0;
  }

  ~Buffer()
  {
    free<T>(_data);
    _count = 0;
  }

  void copy(const T*, const T*, size_t);

  void fill(const T&, size_t, size_t);

  void zero()
  {
    deviceSet(_data, 0, _count * sizeof(T));
  }

  Buffer<T>& operator =(Buffer<T>&& other)
  {
    free<T>(_data);
    _data = other._data;
    _count = other._count;
    other._data = nullptr;
    other._count = 0;
    return *this;
  }

  auto size() const
  {
    return _count;
  }

  operator const value_type*() const
  {
    return _data;
  }

  operator value_type*()
  {
    return _data;
  }

  void dump(const char* s, std::ostream& os = std::cout) const
  {
    cuda::dump<T>(s, _data, _count, os);
  }

private:
  value_type* _data{};
  size_t _count{};

}; // Buffer

template <typename T>
void
Buffer<T>::copy(const T* hBegin, const T* hEnd, size_t offset)
{
  uint32_t length = static_cast<uint32_t>(hEnd - hBegin);

  if (offset + length > _count)
    throw std::logic_error("cuda::Buffer copy: bad range");
  copyToDevice<T>(_data + offset, hBegin, length);
}

template <typename T>
void
Buffer<T>::fill(const T& value, size_t offset, size_t length)
{
  if (offset + length > _count)
    throw std::logic_error("cuda::Buffer fill: bad range");

  T* hData = new T[length];

  std::fill(hData, hData + length, value);
  copyToDevice<T>(_data + offset, hData, length);
  delete []hData;
}


//////////////////////////////////////////////////////////
//
// SoA: CUDA SoA class
// ===
template <typename... Args>
class SoA: public cg::SoA<ArrayAllocator, Args...>
{
public:
  using type = SoA<Args...>;
  using cg::SoA<ArrayAllocator, Args...>::SoA;

}; // SoA

} // end namespace cuda

} // end namespace cg

#endif // __CUDAHelper_h
