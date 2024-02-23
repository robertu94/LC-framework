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
static unsigned int h_QUANT_REL_R_f64_hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}


static inline double h_QUANT_REL_R_f64_log2approx(const double orig_f)
{
  const int mantissabits = 52;
  const long long orig_i = *((long long*)&orig_f);
  const int expo = (orig_i >> mantissabits) & 0x7ff;
  const long long frac_i = (1023LL << mantissabits) | (orig_i & ~(~0ULL << mantissabits));
  const double frac_f = *((double*)&frac_i);
  const double log_f = frac_f + (expo - 1024);  // - bias - 1
  return log_f;
}


static inline double h_QUANT_REL_R_f64_pow2approx(const double log_f)
{
  const int mantissabits = 52;
  const double biased = log_f + 1023;
  const long long expo = biased;
  const double frac_f = biased - (expo - 1);
  const long long frac_i = *((long long*)&frac_f);
  const long long exp_i = (expo << mantissabits) | (frac_i & ~(~0ULL << mantissabits));
  const double recon_f = *((double*)&exp_i);
  return recon_f;
}


static inline void h_QUANT_REL_R_f64(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {throw std::runtime_error("QUANT_REL_R_f64 ERROR: size of input must be a multiple of " + std::to_string(sizeof(double)) + " bytes\n");}
  const int len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {throw std::runtime_error("USAGE: QUANT_REL_R_f64(error_bound [, threshold])\n");}
  const double errorbound = paramv[0];
  const double threshold = (paramc == 2) ? paramv[1] : std::numeric_limits<double>::infinity();
  if (errorbound < 1E-7) {throw std::runtime_error("QUANT_REL_R_f64 ERROR: error_bound must be at least " + std::to_string(1E-7) + "\n");}  // log and exp are too inaccurate below this error bound
  if (threshold <= errorbound) {throw std::runtime_error("QUANT_REL_R_f64 ERROR: threshold must be larger than error_bound\n");}

  long long* const data_i = (long long*)data;

  const int mantissabits = 52;
  const long long signexpomask = ~0ULL << mantissabits;
  const long long maxbin = (1LL << (mantissabits - 2)) - 1;  // leave 2 bits for 2 signs (plus one element)
  const double log2eb = h_QUANT_REL_R_f64_log2approx(1 + errorbound);
  const double inv_log2eb = 1 / log2eb;
  const long long mask = (1LL << mantissabits) - 1;
  const double inv_mask = 1.0 / mask;

  int count = 0;
  #pragma omp parallel for default(none) shared(signexpomask, len, data_i, errorbound, log2eb, inv_log2eb, threshold, maxbin, mantissabits, mask, inv_mask) reduction(+: count)
  for (int i = 0; i < len; i++) {
    const long long orig_i = data_i[i];
    const long long abs_orig_i = orig_i & 0x7fff'ffff'ffff'ffffLL;
    const double abs_orig_f = *((double*)&abs_orig_i);
    long long output = orig_i;
    const int expo = (orig_i >> mantissabits) & 0x7ff;
    if (expo == 0) {  // zero or de-normal values
      if (abs_orig_i == 0) {  // zero
        output = signexpomask | 1;
      }
    } else {
      if (expo == 0x7ff) {  // INF or NaN
        if (((orig_i & signexpomask) == signexpomask) && ((orig_i & ~signexpomask) != 0)) {  // negative NaN
          output = abs_orig_i;  // make positive NaN
        }
      } else {  // normal value
        const double log_f = h_QUANT_REL_R_f64_log2approx(abs_orig_f);
        const double scaled = log_f * inv_log2eb;
        long long bin = (long long)roundf(scaled);
        const long long rnd1 = h_QUANT_REL_R_f64_hash(bin + i + 37);
        const long long rnd2 = h_QUANT_REL_R_f64_hash((bin >> 32) - i - 37);
        const double rnd = inv_mask * (((rnd2 << 32) | rnd1) & mask) - 0.5;  // random noise
        const double abs_recon_f = h_QUANT_REL_R_f64_pow2approx((bin + rnd) * log2eb);
        const double lower = abs_orig_f / (1 + errorbound);
        const double upper = abs_orig_f * (1 + errorbound);
        if ((bin >= maxbin) || (bin <= -maxbin) || (abs_orig_f >= threshold) || (abs_recon_f < lower) || (abs_recon_f > upper) || (abs_recon_f == 0) || !std::isfinite(abs_recon_f)) {
          count++;  // informal only
        } else {
          bin = (bin << 1) ^ (bin >> 63);  // TCMS encoding
          bin = (bin + 1) << 1;
          if (orig_i < 0) bin |= 1;  // include sign
          output = signexpomask | bin;  // 'sign' and 'exponent' fields are all ones, 'mantissa' is non-zero (looks like a negative NaN)
        }
      }
    }
    data_i[i] = (output ^ signexpomask) - 1;
  }

  if (count != 0) printf("QUANT_REL_R_f64: encountered %d non-quantizable values (%.3f%%)\n", count, 100.0 * count / len);  // informal only
}


static inline void h_iQUANT_REL_R_f64(int& size, byte*& data, const int paramc, const double paramv [])
{
  if (size % sizeof(double) != 0) {throw std::runtime_error("QUANT_REL_R_f64 ERROR: size of input must be a multiple of " + std::to_string(sizeof(double)) + " bytes\n");}
  const int len = size / sizeof(double);
  if ((paramc != 1) && (paramc != 2)) {throw std::runtime_error("USAGE: QUANT_REL_R_f64(error_bound [, threshold])\n");}
  const double errorbound = paramv[0];
  if (errorbound < 1E-7) {throw std::runtime_error("QUANT_REL_R_f64 ERROR: error_bound must be at least " + std::to_string(1E-7) + "\n");}  // log and exp are too inaccurate below this error bound

  double* const data_f = (double*)data;
  long long* const data_i = (long long*)data;

  const int mantissabits = 52;
  const long long signexpomask = ~0ULL << mantissabits;
  const double log2eb = h_QUANT_REL_R_f64_log2approx(1 + errorbound);
  const long long mask = (1LL << mantissabits) - 1;
  const double inv_mask = 1.0 / mask;

  #pragma omp parallel for default(none) shared(len, data_f, data_i, log2eb, signexpomask, mask, inv_mask)
  for (int i = 0; i < len; i++) {
    const long long val = (data_i[i] + 1) ^ signexpomask;
    if (((val & signexpomask) == signexpomask) && ((val & ~signexpomask) != 0)) {  // is encoded value
      if (val == (signexpomask | 1)) {
        data_i[i] = 0;
      } else {
        const long long dec = ((val & ~signexpomask) >> 1) - 1;
        const long long bin = (dec >> 1) ^ (((dec << 63) >> 63));  // TCMS decoding
        const long long rnd1 = h_QUANT_REL_R_f64_hash(bin + i + 37);
        const long long rnd2 = h_QUANT_REL_R_f64_hash((bin >> 32) - i - 37);
        const double rnd = inv_mask * (((rnd2 << 32) | rnd1) & mask) - 0.5;  // random noise
        const double abs_recon_f = h_QUANT_REL_R_f64_pow2approx((bin + rnd) * log2eb);
        data_f[i] = (val & 1) ? -abs_recon_f : abs_recon_f;
      }
    } else {
      data_i[i] = val;
    }
  }
}
