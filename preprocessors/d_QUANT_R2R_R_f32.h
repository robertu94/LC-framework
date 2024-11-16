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


#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>


// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static __device__ unsigned int d_QUANT_R2R_R_f32_hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}


static __global__ void d_QUANT_R2R_R_f32_kernel(const int len, byte* const __restrict__ data, byte* const __restrict__ orig_data, const float errorbound, const float* maxf, const float* minf, const float threshold)
{
  float* const orig_data_f = (float*)orig_data;
#ifdef DEBUG
  int* const orig_data_i = (int*)orig_data;
#endif
  float* const data_f = (float*)data;
  int* const data_i = (int*)data;

  const float adj_eb = (*maxf - *minf) * errorbound;
  const int mantissabits = 23;
  const int maxbin = 1 << (mantissabits - 1);  // leave 1 bit for sign
  const float inv_eb = 1 / adj_eb;
  const int mask = (1 << mantissabits) - 1;
  const float inv_mask = 1.0f / mask;

  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    const float orig_f = orig_data_f[idx];
    const float scaled = orig_f * inv_eb;
    const int bin = (int)roundf(scaled);
    const float rnd = inv_mask * (d_QUANT_R2R_R_f32_hash(idx + len) & mask) - 0.5f;  // random noise
    const float recon = (bin + rnd) * adj_eb;

    if ((bin >= maxbin) || (bin <= -maxbin) || (fabsf(orig_f) >= threshold) || (recon < orig_f - adj_eb) || (recon > orig_f + adj_eb) || (fabsf(orig_f - recon) > adj_eb) || (orig_f != orig_f)) {  // last check is to handle NaNs
      data_f[idx] = orig_f;
#ifdef DEBUG
      assert(((orig_data_i[idx] >> mantissabits) & 0xff) != 0);
#endif
    } else {
      data_i[idx] = (bin << 1) ^ (bin >> 31);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
  }

  if (idx == 0) {
    data_f[len] = adj_eb;
  }
}


static __global__ void d_iQUANT_R2R_R_f32_kernel(const int len, byte* const __restrict__ data)
{
  float* const data_f = (float*)data;
  int* const data_i = (int*)data;

  const float errorbound = data_f[len];
  const int mantissabits = 23;
  const int mask = (1 << mantissabits) - 1;
  const float inv_mask = 1.0f / mask;

  const int idx = threadIdx.x + blockIdx.x * TPB;
  if (idx < len) {
    int bin = data_i[idx];
    if ((0 <= bin) && (bin < (1 << mantissabits))) {  // is encoded value
      bin = (bin >> 1) ^ (((bin << 31) >> 31));  // TCMS decoding
      const float rnd = inv_mask * (d_QUANT_R2R_R_f32_hash(idx + len) & mask) - 0.5f;  // random noise
      data_f[idx] = (bin + rnd) * errorbound;
    }
  }
}


static inline void d_QUANT_R2R_R_f32(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(float) != 0) {throw std::runtime_error("QUANT_R2R_R_f32: ERROR: size of input must be a multiple of " + std::to_string(sizeof(float)) + " bytes\n");}
  const int len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {throw std::runtime_error("USAGE: QUANT_R2R_R_f32(error_bound [, threshold])\n");}
  const float errorbound = paramv[0];
  const float threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<float>::infinity();
  if (errorbound < std::numeric_limits<float>::min()) {throw std::runtime_error("QUANT_R2R_R_f32: ERROR: error_bound must be at least " + std::to_string(std::numeric_limits<float>::min()) + "\n");}  // minimum positive normalized value
  if (threshold <= errorbound) {throw std::runtime_error("QUANT_R2R_R_f32: ERROR: threshold must be larger than error_bound\n");}

  byte* d_new_data;
  if (cudaSuccess != cudaMalloc((void**) &d_new_data, size + sizeof(float))) {
    fprintf(stderr, "ERROR: could not allocate d_new_data\n\n");
    throw std::runtime_error("LC error");
  }

  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast((float*)data);
  thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> min_max = thrust::minmax_element(thrust::device, dev_ptr, dev_ptr + len);

  d_QUANT_R2R_R_f32_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, d_new_data, data, errorbound, thrust::raw_pointer_cast(min_max.second), thrust::raw_pointer_cast(min_max.first), threshold);

  cudaFree(data);
  data = (byte*) d_new_data;
  size += sizeof(float);
}


static inline void d_iQUANT_R2R_R_f32(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(float) != 0) {throw std::runtime_error("QUANT_R2R_R_f32: ERROR: size of input must be a multiple of " + std::to_string(sizeof(float)) + " bytes\n");}
  const int len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {throw std::runtime_error("USAGE: QUANT_R2R_R_f32(error_bound [, threshold])\n");}

  d_iQUANT_R2R_R_f32_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len - 1, data);

  size -= sizeof(float);
}
