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


// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static unsigned int h_QUANT_ABS_R_f32_hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}


static inline void h_QUANT_ABS_R_f32(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(float) != 0) {throw std::runtime_error("QUANT_ABS_R_f32: ERROR: size of input must be a multiple of " + std::to_string(sizeof(float)) + " bytes\n");}
  const int len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {throw std::runtime_error("USAGE: QUANT_ABS_R_f32(error_bound [, threshold])\n");}
  const float errorbound = paramv[0];
  const float threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<float>::infinity();
  if (errorbound < std::numeric_limits<float>::min()) {throw std::runtime_error("QUANT_ABS_R_f32: ERROR: error_bound must be at least " + std::to_string(std::numeric_limits<float>::min()) + "\n");}  // minimum positive normalized value
  if (threshold <= errorbound) {throw std::runtime_error("QUANT_ABS_R_f32: ERROR: threshold must be larger than error_bound\n");}

  float* const data_f = (float*)data;
  int* const data_i = (int*)data;

  const int mantissabits = 23;
  const int maxbin = 1 << (mantissabits - 1);  // leave 1 bit for sign
  const float inv_eb = 1 / errorbound;
  const int mask = (1 << mantissabits) - 1;
  const float inv_mask = 1.0f / mask;

  int count = 0;
  #pragma omp parallel for default(none) shared(len, data_f, data_i, errorbound, inv_eb, threshold, maxbin, mask, inv_mask, mantissabits) reduction(+: count)
  for (int i = 0; i < len; i++) {
    const float orig_f = data_f[i];
    const float scaled = orig_f * inv_eb;
    const int bin = (int)roundf(scaled);
    const float rnd = inv_mask * (h_QUANT_ABS_R_f32_hash(i + len) & mask) - 0.5f;  // random noise
    const float recon = (bin + rnd) * errorbound;

    if ((bin >= maxbin) || (bin <= -maxbin) || (fabsf(orig_f) >= threshold) || (recon < orig_f - errorbound) || (recon > orig_f + errorbound) || (fabsf(orig_f - recon) > errorbound) || (orig_f != orig_f)) {  // last check is to handle NaNs
      count++;  // informal only
      assert(((data_i[i] >> mantissabits) & 0xff) != 0);
    } else {
      data_i[i] = (bin << 1) ^ (bin >> 31);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
  }

  if (count != 0) printf("QUANT_ABS_R_f32: encountered %d non-quantizable values (%.3f%%)\n", count, 100.0 * count / len);  // informal only
}


static inline void h_iQUANT_ABS_R_f32(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(float) != 0) {throw std::runtime_error("QUANT_ABS_R_f32: ERROR: size of input must be a multiple of " + std::to_string(sizeof(float)) + " bytes\n");}
  const int len = size / sizeof(float);
  if ((paramc != 1) && (paramc != 2)) {throw std::runtime_error("USAGE: QUANT_ABS_R_f32(error_bound [, threshold])\n");}
  const float errorbound = paramv[0];
  if (errorbound < std::numeric_limits<float>::min()) {throw std::runtime_error("QUANT_ABS_R_f32: ERROR: error_bound must be at least " + std::to_string(std::numeric_limits<float>::min()) + "\n");}  // minimum positive normalized value

  float* const data_f = (float*)data;
  int* const data_i = (int*)data;

  const int mantissabits = 23;
  const int mask = (1 << mantissabits) - 1;
  const float inv_mask = 1.0f / mask;

  #pragma omp parallel for default(none) shared(len, data_f, data_i, errorbound, mantissabits, mask, inv_mask)
  for (int i = 0; i < len; i++) {
    int bin = data_i[i];
    if ((0 <= bin) && (bin < (1 << mantissabits))) {  // is encoded value
      bin = (bin >> 1) ^ (((bin << 31) >> 31));  // TCMS decoding
      const float rnd = inv_mask * (h_QUANT_ABS_R_f32_hash(i + len) & mask) - 0.5f;  // random noise
      data_f[i] = (bin + rnd) * errorbound;
    }
  }
}
