#include <ATen/ATen.h>
#include "compat.h"

// Forward/backward compatiblity hack around
// https://github.com/pytorch/pytorch/commit/3aeb78079bcd68282fe9117088e138b77318e288
// pending more future-proof guidance from upstream.
// struct TypeShim
// {
//   const at::Type& payload;
//   TypeShim(const at::Type& type) : payload(type) {}
//   // Enable trivial conversion to a const at::Type& for pre-3aeb78
//   operator const at::Type&(){ return payload; };
//   // Enable dispatch switch statements to take *this directly for  post-3aeb78
//   //operator at::ScalarType(){ return payload.; };
// };

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...) \
  switch(TYPE) \
  { \
    case at::ScalarType::Float: \
    { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: \
    { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
  }


#define DISPATCH_FLOAT_HALF_AND_BFLOAT(TYPE, LEVEL, NAME, ...) \
  switch(TYPE) \
  { \
    case at::ScalarType::Float: \
    { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: \
    { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::BFloat16: \
    { \
      using scalar_t_##LEVEL = at::BFloat16; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
  }


#define DISPATCH_FLOAT_HALF_AND_BYTE(TYPE, LEVEL, NAME, ...) \
  switch(TYPE) \
  { \
    case at::ScalarType::Float: \
    { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: \
    { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Byte: \
    { \
      using scalar_t_##LEVEL = uint8_t; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
  }


#define DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...) \
  switch(TYPE) \
  { \
    case at::ScalarType::Double: \
    { \
      using scalar_t_##LEVEL = double; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Float: \
    { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: \
    { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
  }


#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(TYPE, LEVEL, NAME, ...) \
  switch(TYPE) \
  { \
    case at::ScalarType::Double: \
    { \
      using scalar_t_##LEVEL = double; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Float: \
    { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: \
    { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::BFloat16: \
    { \
      using scalar_t_##LEVEL = at::BFloat16; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
  }


  #define DISPATCH_DOUBLE_AND_FLOAT(TYPE, LEVEL, NAME, ...) \
  switch(TYPE) \
  { \
    case at::ScalarType::Double: \
    { \
      using scalar_t_##LEVEL = double; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Float: \
    { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
  }


  #define DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, ...)			\
  switch(TYPE)								\
    {									\
    case at::ScalarType::Half:						\
      {									\
	using scalar_t = at::Half;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    case at::ScalarType::BFloat16:					\
      {									\
	using scalar_t = at::BFloat16;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    default:								\
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");	\
  }


  #define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch(TYPEIN)							\
    {									\
    case at::ScalarType::Float:						\
      {									\
	using scalar_t_in = float;					\
	switch(TYPEOUT)							\
	  {								\
	  case at::ScalarType::Float:					\
	    {								\
	      using scalar_t_out = float;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::Half:					\
	    {								\
	      using scalar_t_out = at::Half;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::BFloat16:				\
	    {								\
	      using scalar_t_out = at::BFloat16;			\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  default:							\
	    AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
	  }								\
	break;								\
      }									\
    case at::ScalarType::Half:						\
      {									\
	using scalar_t_in = at::Half;					\
	using scalar_t_out = at::Half;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    case at::ScalarType::BFloat16:					\
      {									\
	using scalar_t_in = at::BFloat16;				\
	using scalar_t_out = at::BFloat16;				\
	__VA_ARGS__;							\
	break;								\
      }									\
    default:								\
      AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");	\
    }


  #define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch(TYPEIN)							\
    {									\
    case at::ScalarType::Double:						\
      {									\
	using scalar_t_in = double;					\
	switch(TYPEOUT)							\
	  {								\
	  case at::ScalarType::Double:					\
	    {								\
	      using scalar_t_out = double;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::Float:					\
	    {								\
	      using scalar_t_out = float;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::Half:					\
	    {								\
	      using scalar_t_out = at::Half;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::BFloat16:				\
	    {								\
	      using scalar_t_out = at::BFloat16;			\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  default:							\
	    AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
	  }								\
	break;								\
      }									\
    case at::ScalarType::Float:						\
      {									\
	using scalar_t_in = float;					\
	switch(TYPEOUT)							\
	  {								\
	  case at::ScalarType::Float:					\
	    {								\
	      using scalar_t_out = float;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::Half:					\
	    {								\
	      using scalar_t_out = at::Half;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::BFloat16:				\
	    {								\
	      using scalar_t_out = at::BFloat16;			\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  default:							\
	    AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
	  }								\
	break;								\
      }									\
    case at::ScalarType::Half:						\
      {									\
	using scalar_t_in = at::Half;					\
	using scalar_t_out = at::Half;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    case at::ScalarType::BFloat16:					\
      {									\
	using scalar_t_in = at::BFloat16;				\
	using scalar_t_out = at::BFloat16;				\
	__VA_ARGS__;							\
	break;								\
      }									\
    default:								\
      AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");	\
    }


template<typename T>
__device__ __forceinline__ T reduce_block_into_lanes
  (T *x,
   T val,
   int lanes=1,
   bool share_result=false) // lanes is intended to be <= warpSize.
{
  int tid = threadIdx.x + threadIdx.y*blockDim.x;
  int blockSize = blockDim.x*blockDim.y; // blockSize is intended to be a multiple of warpSize.
  auto double_warp_size = warpSize * 2;

  if(blockSize >= double_warp_size)
  {
    x[tid] = val;
    __syncthreads();
  }

  #pragma unroll
  for(int i = (blockSize >> 1); i >= double_warp_size; i >>= 1)
  {
    if(tid < i)
      x[tid] = x[tid] + x[tid+i];
    __syncthreads();
  }

  T final;

  if(tid < warpSize)
  {
    if(blockSize >= double_warp_size)
      final = x[tid] + x[tid + warpSize];
    else
      final = val;
    // __SYNCWARP();

    #pragma unroll
    for(int i = warpSize / 2; i >= lanes; i >>= 1)
      final = final + __shfl_down_sync(0xffffffff, final, i);
  }

  if(share_result)
  {
    if(tid < lanes)
      x[tid] = final; // EpilogueOp
    // Make sure the smem result is visible to all warps.
    // __syncthreads();
  }
  __syncthreads();

  return final;
}

template<typename T>
__device__ __forceinline__ T reduce_block_into_lanes_max_op
  (T *x,
   T val,
   int lanes=1,
   bool share_result=false) // lanes is intended to be <= warpSize.
{
  int tid = threadIdx.x + threadIdx.y*blockDim.x;
  int blockSize = blockDim.x*blockDim.y; // blockSize is intended to be a multiple of warpSize.
  auto double_warp_size = warpSize * 2;

  if(blockSize >= double_warp_size)
  {
    x[tid] = val;
    __syncthreads();
  }

  #pragma unroll
  for(int i = (blockSize >> 1); i >= double_warp_size; i >>= 1)
  {
    if(tid < i)
      x[tid] = fmaxf(fabsf(x[tid]), fabsf(x[tid+i]));
    __syncthreads();
  }

  T final;

  if(tid < warpSize)
  {
    if(blockSize >= double_warp_size)
      final = fmaxf(fabsf(x[tid]), fabsf(x[tid + warpSize]));
    else
      final = val;
    // __SYNCWARP();

    #pragma unroll
    for(int i = warpSize / 2; i >= lanes; i >>= 1)
      final = fmaxf(fabsf(final), fabsf(__shfl_down_sync(0xffffffff, final, i)));
  }

  if(share_result)
  {
    if(tid < lanes)
      x[tid] = final; // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}
