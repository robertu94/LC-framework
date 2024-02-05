/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2024, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#include "include/d_repetition_elimination.h"


static __device__ inline bool d_RRE_2(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  using type = unsigned short;
  const int tid = threadIdx.x;
  const int size = csize / sizeof(type);  // words in chunk (rounded down)
  const int extra = csize % sizeof(type);
  const int avail = CS - 2 - extra;
  const int bits = 8 * sizeof(type);
  assert(CS == 16384);

  // zero out bitmap
  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (csize < CS) {
    for (int i = csize / bits + tid; i < CS / bits; i += TPB) {
      bitmap[i] = 0;
    }
    __syncthreads();
  }

  // copy non-repeating values and generate bitmap
  int wpos = 0;
  if (size > 0) d_REencode((type*)in, size, (type*)out, wpos, (type*)bitmap, temp_w);
  wpos *= sizeof(type);
  if (wpos >= avail) return false;
  __syncthreads();

  // check if not all zeros
  if (wpos != 0) {
    // iteratively compress bitmaps
    int base = 0 / sizeof(type);
    int range = 2048 / sizeof(type);
    int cnt = avail - wpos;
    if (!d_REencode<byte, 2048 / sizeof(type), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = 2048 / sizeof(type);
    range = 256 / sizeof(type);
    cnt = avail - wpos;
    if (!d_REencode<byte, 256 / sizeof(type), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = (2048 + 256) / sizeof(type);
    range = 32 / sizeof(type);
    if constexpr (sizeof(type) < 8) {
      cnt = avail - wpos;
      if (!d_REencode<byte, 32 / sizeof(type), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
      wpos += cnt;

      base = (2048 + 256 + 32) / sizeof(type);
      range = 4 / sizeof(type);
    }

    // output last level of bitmap
    if (wpos >= avail - range) return false;
    if (tid < range) {  // 4 / sizeof(type)
      out[wpos + tid] = bitmap[base + tid];
    }
    wpos += range;
  }

  // copy leftover bytes
  if constexpr (sizeof(type) > 1) {
    if (tid < extra) out[wpos + tid] = in[csize - extra + tid];
  }

  // output old csize and update csize
  const int new_size = wpos + 2 + extra;
  if (tid == 0) {
    out[new_size - 2] = csize;  // bottom byte
    out[new_size - 1] = csize >> 8;  // second byte
  }
  csize = new_size;
  return true;
}


static __device__ inline void d_iRRE_2(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  using type = unsigned short;
  const int tid = threadIdx.x;
  int rpos = csize;
  csize = (int)in[--rpos] << 8;  // second byte
  csize |= in[--rpos];  // bottom byte
  const int size = csize / sizeof(type);  // words in chunk (rounded down)
  assert(CS == 16384);
  assert(TPB >= 256);

  // copy leftover byte
  if constexpr (sizeof(type) > 1) {
    const int extra = csize % sizeof(type);
    if (tid < extra) out[csize - extra + tid] = in[rpos - extra + tid];
    rpos -= extra;
  }

  if (rpos == 0) {
    // all zeros
    type* const out_t = (type*)out;
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = 0;
    }
  } else {
    int* const temp_w = (int*)temp;
    byte* const bitmap = (byte*)&temp_w[WS];

    // iteratively decompress bitmaps
    int base, range;
    if constexpr (sizeof(type) == 8) {
      base = (2048 + 256) / sizeof(type);
      range = 32 / sizeof(type);
      // read in last level of bitmap
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];
    } else {
      base = (2048 + 256 + 32) / sizeof(type);
      range = 4 / sizeof(type);
      // read in last level of bitmap
      rpos -= range;
      if (tid < range) bitmap[base + tid] = in[rpos + tid];

      rpos -= __syncthreads_count((tid < range * 8) && ((in[rpos + tid / 8] >> (tid % 8)) & 1));
      base = (2048 + 256) / sizeof(type);
      range = 32 / sizeof(type);
      d_REdecode<byte, 32 / sizeof(type)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    }
    __syncthreads();

    rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    base = 2048 / sizeof(type);
    range = 256 / sizeof(type);
    d_REdecode<byte, 256 / sizeof(type)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    if constexpr (sizeof(type) >= 4) {
      rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    }
    if constexpr (sizeof(type) == 2) {
      int sum = __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
      sum += __syncthreads_count((tid + TPB < range * 8) && ((bitmap[base + (tid + TPB) / 8] >> (tid % 8)) & 1));
      rpos -= sum;
    }
    if constexpr (sizeof(type) == 1) {
      int sum = 0;
      for (int i = 0; i < TPB * 4; i += TPB) {
        sum += __syncthreads_count((tid + i < range * 8) && ((bitmap[base + (tid + i) / 8] >> (tid % 8)) & 1));
      }
      rpos -= sum;
    }
    base = 0 / sizeof(type);
    range = 2048 / sizeof(type);
    d_REdecode<byte, 2048 / sizeof(type)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // copy non-repeating values based on bitmap
    if (size > 0) d_REdecode(size, (type*)in, (type*)bitmap, (type*)out, temp_w);
  }
}
