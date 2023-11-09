/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2023, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
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

---
This file is adapted from the code of LibPressio

Copyright © 2022 , UChicago Argonne, LLC
All Rights Reserved
[libpressio, Version 0.97.3]
Robert Underwood
Argonne National Laboratory

OPEN SOURCE LICENSE (license number: SF-19-112)
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.  Software changes, modifications, or derivative works, should be noted with comments and the author and organization's name.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
3. Neither the names of UChicago Argonne, LLC or the Department of Energy nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 
4. The software and the end-user documentation included with the redistribution, if any, must include the following acknowledgment:
 
   "This product includes software produced by UChicago Argonne, LLC under Contract No. DE-AC02-06CH11357 with the Department of Energy."
 
******************************************************************************************************
DISCLAIMER
 
THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.
 
NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
 
***************************************************************************************************

Contact: Sheng Di (sdi1@anl.gov), Robert Underwood (runderwood@anl.gov)
*/

using byte = unsigned char;
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <utility>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <map>
#include <chrono>
#include <regex>

#include "verifiers/include/verifiers.h"
#include "include/consts.h"
#ifndef USE_GPU
  #ifndef USE_CPU
  //no CPU and no GPU
  #else
  #include "preprocessors/include/CPUpreprocessors.h"
  #include "components/include/CPUcomponents.h"
  #endif
#else
  #include <cuda.h>
  #include "include/max_reduction.h"
  #include "include/max_scan.h"
  #include "include/prefix_sum.h"
  #include "include/sum_reduction.h"

  #ifndef USE_CPU
  #include "preprocessors/include/GPUpreprocessors.h"
  #include "components/include/GPUcomponents.h"
  #else
  #include "preprocessors/include/preprocessors.h"
  #include "components/include/components.h"
  #endif
#endif

namespace libpressio { namespace lc_ns {

static const int max_stages = 8;  // cannot be more than 8
static std::vector<byte> getStages(std::map<std::string, byte> comp_name2num, std::vector<std::string> const& entries)
{
  std::vector<byte> comp_list;
  std::transform(entries.begin(), entries.end(), std::back_inserter(comp_list), [&comp_name2num](std::string const& e){
    return comp_name2num.at(e);
  });
  return comp_list;
}

static std::vector<std::pair<byte, std::vector<double>>> getItems(std::map<std::string, byte> item_name2num, std::vector<std::string> const& names)
{
  std::vector<std::pair<byte, std::vector<double>>> items;

  for (auto i : names) {
    // get name
    const char* p = i.c_str();
    const char* beg = i.c_str();
    while ((*p != 0) && (*p != ' ') && (*p != '\t') && (*p != '(')) p++;  // find end of name
    const char* end = i.c_str();
    if (end <= beg) {fprintf(stderr, "ERROR: expected an item name in specification\n\n"); exit(-1);}
    int num = -1;
    for (auto pair: item_name2num) {
      const std::string itemname = pair.first;
      const byte itemnum = pair.second;
      if (i.substr(0, end-beg) == itemname) {
        num = itemnum;
        break;
      }
    }
    if (num < 0) {fprintf(stderr, "ERROR: unknown item name\n\n"); exit(-1);}

    // read in parameters
    std::vector<double> params;
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    if (*p != '(') {fprintf(stderr, "ERROR: expected '(' in specification\n\n"); exit(-1);}
    p++;
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    while ((*p != 0) && (*p != ')')) {
      // get double
      char* pos;
      const double d = std::strtod(p, &pos);
      if (pos == p) {fprintf(stderr, "ERROR: expected a value in specification\n\n"); exit(-1);}
      p = pos;
      params.push_back(d);
      while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
      if (*p == ')') break;

      // consume comma
      if (*p != ',') {fprintf(stderr, "ERROR: expected ',' in specification\n\n"); exit(-1);}
      p++;
      while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
    }
    if (*p != ')') {fprintf(stderr, "ERROR: expected ')' in specification\n\n"); exit(-1);}
    p++;
    items.push_back(std::make_pair((byte)num, params));
    while ((*p != 0) && ((*p == ' ') || (*p == '\t'))) p++;  // skip over white space
  }

  return items;
}

static void verify(const int size, const byte* const recon, const byte* const orig, std::vector<std::pair<byte, std::vector<double>>> verifs)
{
  for (int i = 0; i < verifs.size(); i++) {
    std::vector<double> params = verifs[i].second;
    switch (verifs[i].first) {
      default: fprintf(stderr, "ERROR: unknown verifier\n\n"); exit(-1); break;
      /*##switch-verify-beg##*/

      // code will be automatically inserted

      /*##switch-verify-end##*/
    }
  }
}

static std::map<std::string, byte> getVerifMap()
{
  std::map<std::string, byte> verifs;
  /*##verifier-map-beg##*/

  // code will be automatically inserted

  /*##verifier-map-end##*/
  return verifs;
}


static std::map<std::string, byte> getPreproMap()
{
  std::map<std::string, byte> preprocessors;
  preprocessors["NUL"] = 0;
  /*##preprocessor-map-beg##*/

  // code will be automatically inserted

  /*##preprocessor-map-end##*/
  return preprocessors;
}


static std::map<std::string, byte> getCompMap()
{
  std::map<std::string, byte> components;
  components["NUL"] = 0;
  /*##component-map-beg##*/

  // code will be automatically inserted

  /*##component-map-end##*/
  return components;
}


static std::string getPipeline(unsigned long long pipeline, const int stages)
{
  std::string s;
  for (int i = 0; i < stages; i++) {
    switch (pipeline & 0xff) {
      default: s += " NUL"; break;
      /*##switch-pipeline-beg##*/

      // code will be automatically inserted

      /*##switch-pipeline-end##*/
    }
    pipeline >>= 8;
  }
  s.erase(0, 1);
  return s;
}

#ifdef USE_CPU
static void h_encode(const unsigned long long chain, const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int& outsize, uint32_t nthreads)
{
  // initialize
  const int chunks = (insize + CS - 1) / CS;  // round up
  int* const head_out = (int*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];
  int* const carry = new int [chunks];
  memset(carry, 0, chunks * sizeof(int));

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const int base = chunkID * CS;
    const int osize = std::min(CS, insize - base);
    memcpy(out, &input[base], osize);

    // encode chunk
    int csize = osize;
    bool good = true;
    unsigned long long pipeline = chain;
    while ((pipeline != 0) && good) {
      std::swap(in, out);
      switch (pipeline & 0xff) {
        default: std::swap(in, out); break;
        /*##switch-host-encode-beg##*/

        // code will be automatically inserted

        /*##switch-host-encode-end##*/
      }
      pipeline >>= 8;
    }

    // handle carry and store chunk
    int offs = 0;
    if (chunkID > 0) {
      do {
        #pragma omp atomic read
        offs = carry[chunkID - 1];
      } while (offs == 0);
      #pragma omp flush
    }
    if (good && (csize < osize)) {
      // store compressed data
      #pragma omp atomic write
      carry[chunkID] = offs + csize;
      size_out[chunkID] = csize;
      memcpy(&data_out[offs], out, csize);
    } else {
      // store original data
      #pragma omp atomic write
      carry[chunkID] = offs + osize;
      size_out[chunkID] = osize;
      memcpy(&data_out[offs], &input[base], osize);
    }
  }

  // output header
  head_out[0] = insize;

  // finish
  outsize = &data_out[carry[chunks - 1]] - output;
  delete [] carry;
}

static void h_decode(const unsigned long long chain, const byte* const __restrict__ input, byte* const __restrict__ output, int& outsize, uint32_t n_threads)
{
  // input header
  int* const head_in = (int*)input;
  outsize = head_in[0];

  // initialize
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];
  int* const start = new int [chunks];

  // convert chunk sizes into starting positions
  int pfs = 0;
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += (int)size_in[chunkID];
  }

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
  for (int chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const int base = chunkID * CS;
    const int osize = std::min(CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      // decompress
      memcpy(out, &data_in[start[chunkID]], csize);

      // decode
      unsigned long long pipeline = chain;
      while (pipeline != 0) {
        std::swap(in, out);
        switch (pipeline >> 56) {
          default: std::swap(in, out); break;
          /*##switch-host-decode-beg##*/

          // code will be automatically inserted

          /*##switch-host-decode-end##*/
        }
        pipeline <<= 8;
      }

      if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d\n\n", csize, osize); exit(-1);}
      memcpy(&output[base], out, csize);
    }
  }

  // finish
  delete [] start;
}

static void h_preprocess_encode(int& hpreencsize, byte*& hpreencdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = 0; i < prepros.size(); i++) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); exit(-1); break;
      /*##switch-host-preprocess-encode-beg##*/

      // code will be automatically inserted

      /*##switch-host-preprocess-encode-end##*/
    }
  }
}


static void h_preprocess_decode(int& hpredecsize, byte*& hpredecdata, std::vector<std::pair<byte, std::vector<double>>> prepros)
{
  for (int i = prepros.size() - 1; i >= 0; i--) {
    std::vector<double> params = prepros[i].second;
    switch (prepros[i].first) {
      default: fprintf(stderr, "ERROR: unknown preprocessor\n\n"); exit(-1); break;
      /*##switch-host-preprocess-decode-beg##*/

      // code will be automatically inserted

      /*##switch-host-preprocess-decode-end##*/
    }
  }
}
#endif

template <class K, class V>
std::vector<K> map_keys(std::map<K,V> const& map) {
        std::vector<K> keys;
        std::transform(map.begin(), map.end(), std::back_inserter(keys), [](auto const& pair){ return pair.first;});
        return keys;
}
template <class K, class V>
std::map<V,K> map_flip(std::map<K,V> const& map) {
        std::map<V,K> flipped;
        std::transform(map.begin(), map.end(), std::back_inserter(flipped), [](auto const& pair){ return std::make_pair(pair.second, pair.first);});
        return flipped;
}

class lc_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "lc:components", component_steps);
    set(options, "lc:preprocessors", preprocessor_steps);
    set(options, "pressio:nthreads", n_threads);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    set(options, "lc:components", map_keys(getCompMap()));
    set(options, "lc:preprocessors", map_keys(getPreproMap()));
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"()");
    set(options, "lc:components", "list of components steps to use with LC");
    set(options, "lc:preprocessors", "list of preprocessors steps to use with LC");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    std::vector<std::string> tmp_comp_steps, tmp_preprocess_steps;
    if(get(options, "lc:components", &tmp_comp_steps) == pressio_options_key_set) {
        if(tmp_comp_steps.size() > max_stages) {
            return set_error(1, "too many component stages");
        }
        component_steps = std::move(tmp_comp_steps);
    }
    if(get(options, "lc:preprocessors", &tmp_preprocess_steps) == pressio_options_key_set) {
        if(tmp_preprocess_steps.size() > max_stages) {
            return set_error(1, "too many encoding stages");
        }
        component_steps = std::move(tmp_preprocess_steps);
    }
    get(options, "pressio:nthreads", &n_threads);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
      auto prepros = getItems(getPreproMap(), preprocessor_steps);
      pressio_data hpreencodedata = pressio_data::clone(*input);
      unsigned char* hprencodedata_ptr = static_cast<unsigned char*>(hpreencodedata.data());
      int hpreencodesize = hpreencodedata.size_in_bytes();
      auto preprocess_begin = std::chrono::steady_clock::now();
      h_preprocess_encode(hpreencodesize, hprencodedata_ptr, prepros);
      auto preprocess_end = std::chrono::steady_clock::now();
      preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end-preprocess_begin).count();
      view_segment(&hpreencodedata, "preprocessed");

      const int hchunks = (hpreencodesize + CS - 1) / CS;  // round up
      const int hmaxsize = 3 * sizeof(int) + hchunks * sizeof(short) + hchunks * CS;  //MB: adjust later
      pressio_data hencoded = pressio_data::owning(pressio_byte_dtype, {static_cast<size_t>(hmaxsize)});
      int hencsize = 0;

      auto encode_begin = std::chrono::steady_clock::now();
      h_encode(get_chain(), static_cast<byte*>(hpreencodedata.data()), hpreencodesize, static_cast<byte*>(hencoded.data()), hencsize, n_threads.value_or(omp_get_num_threads()));
      auto encode_end = std::chrono::steady_clock::now();
      encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(encode_end-encode_begin).count();
      hencoded.set_dimensions({static_cast<size_t>(hencsize)});
      *output = std::move(hencoded);

      return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
      int hdecsize = 0;
      unsigned long long chain = get_chain();
      unsigned long long schain = chain;
      if (chain != 0) {
        while ((schain >> 56) == 0) schain <<= 8;
      }
      
      int hpreencsize;
      memcpy(&hpreencsize, input->data(), sizeof(hdecsize));
      pressio_data hdecoded = pressio_data::owning(pressio_byte_dtype, {static_cast<size_t>(hpreencsize)});
      auto encode_begin = std::chrono::steady_clock::now();
      h_decode(schain, static_cast<const byte*>(input->data()), static_cast<byte*>(hdecoded.data()), hdecsize, n_threads.value_or(omp_get_num_threads()));
      auto encode_end = std::chrono::steady_clock::now();
      decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(encode_end-encode_begin).count();

      auto prepros = getItems(getPreproMap(), preprocessor_steps);
      int houtputsize = output->size_in_bytes();
      byte* data = static_cast<byte*>(hdecoded.data());
      auto preprocess_begin = std::chrono::steady_clock::now();
      h_preprocess_decode(houtputsize, data, prepros);
      auto preprocess_end = std::chrono::steady_clock::now();
      preprocess_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end-preprocess_begin).count();
      hdecoded.set_dtype(output->dtype());
      auto dims = output->dimensions();
      hdecoded.set_dimensions(std::move(dims));
      *output = std::move(hdecoded);
      return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "lc"; }

  pressio_options get_metrics_results_impl() const override {
      pressio_options options;
      set(options, "lc:preprocess_time", preprocess_ms);
      set(options, "lc:encode_time", encode_ms);
      set(options, "lc:preprocess_decode_time", preprocess_decode_ms);
      set(options, "lc:decode_time", decode_ms);
      return options;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<lc_compressor_plugin>(*this);
  }

private:
  unsigned long long get_chain() {
      unsigned long long chain = 0;
      std::vector<byte> comp_list;
      comp_list = getStages(getCompMap(), component_steps);
      for (int s = 0; s < comp_list.size(); s++) {
        unsigned long long compnum = comp_list[s];
        chain |= compnum << (s * 8);
      }
      return chain;
  }

  std::vector<std::string> component_steps;
  std::vector<std::string> preprocessor_steps;
  uint64_t preprocess_ms=0, encode_ms=0, decode_ms=0, preprocess_decode_ms=0;
  compat::optional<uint32_t> n_threads;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "lc", []() {
  return compat::make_unique<lc_compressor_plugin>();
});

} }

extern "C" void libpressio_register_lc() {
}
