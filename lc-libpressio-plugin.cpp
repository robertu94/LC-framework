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
#include "lc.h"

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

static std::vector<byte> getStagesSingleCompressor(std::map<std::string, byte> comp_name2num, std::vector<std::string> const& entries)
{
  std::vector<byte> comp_list;
  std::transform(entries.begin(), entries.end(), std::back_inserter(comp_list), [&comp_name2num](std::string const& e){
    return comp_name2num.at(e);
  });
  return comp_list;
}

static std::vector<std::pair<byte, std::vector<double>>> getItemsSingleCompressor(std::map<std::string, byte> item_name2num, std::vector<std::string> const& names)
{
    std::vector<std::pair<byte, std::vector<double>>> items;

    std::regex spec_regex("([a-zA-Z0-9_]+)(?:\\(([^,]+(?:,(?:[^,]+))*)\\))?");
    std::regex params_regex("[^,]+");
    std::smatch spec_match;
    for (auto name : names) {
        if(!std::regex_match(name, spec_match, spec_regex)) {
            throw std::runtime_error("invalid item specficiation");
        }
        std::string itemname = spec_match[1].str();
        int num = item_name2num.at(itemname);

        // read in parameters
        std::vector<double> params;
        if(spec_match[2].matched) {
            std::sregex_iterator params_begin(spec_match[2].first, spec_match[2].second, params_regex);
            std::sregex_iterator params_end;
            std::transform(params_begin, params_end, std::back_inserter(params), [](std::smatch const& it){
                        return std::stod(it.str());
                    });
        }
        items.emplace_back(std::move(num), std::move(params));
    }

    return items;
}







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
    set(options, "pressio:highlevel", std::vector<std::string>{"pressio:nthreads","lc:components", "lc:preprocessors"});
    set(options, "predictors:runtime", std::vector<std::string>{"pressio:nthreads","lc:components", "lc:preprocessors"});
    set(options, "predictors:error_agnostic", std::vector<std::string>{"lc:components", "lc:preprocessors"});
    set(options, "predictors:error_dependent", std::vector<std::string>{"lc:preprocessors"});
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"()");
    set(options, "lc:components", "list of components steps to use with LC");
    set(options, "lc:preprocessors", "list of preprocessors steps to use with LC");
    set(options, "lc:preprocess_time", "preprocessing time in milliseconds");
    set(options, "lc:encode_time", "encoding time in milliseconds");
    set(options, "lc:preprocess_decode_time", "preprocessing decodeing time in milliseconds");
    set(options, "lc:decode_time", "decoding time in milliseconds");
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
        preprocessor_steps = std::move(tmp_preprocess_steps);
    }
    get(options, "pressio:nthreads", &n_threads);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
      try {
          auto prepros = getItemsSingleCompressor(getPreproMap(), preprocessor_steps);

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
      } catch (std::runtime_error const& ex) {
          return set_error(1, std::string("invalid stage: ") + ex.what());
      } catch (std::out_of_range const& ex) {
          return set_error(2, std::string("invalid preprocessor: ") + ex.what());
      } catch (std::invalid_argument const& ex) {
          return set_error(3, std::string("invalid preprocessor argument: ") + ex.what());
      }

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

      try {
          auto prepros = getItemsSingleCompressor(getPreproMap(), preprocessor_steps);
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
      } catch (std::runtime_error const& ex) {
          return set_error(1, std::string("invalid stage: ") + ex.what());
      } catch (std::out_of_range const& ex) {
          return set_error(2, std::string("invalid preprocessor: ") + ex.what());
      } catch (std::invalid_argument const& ex) {
          return set_error(3, std::string("invalid preprocessor argument: ") + ex.what());
      }
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
      comp_list = getStagesSingleCompressor(getCompMap(), component_steps);
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
