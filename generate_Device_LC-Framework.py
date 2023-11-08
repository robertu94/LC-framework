#!/usr/bin/env python3

"""
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
"""

import re
import glob, os
from pathlib import Path
import math
import shutil

# generate lc framework
shutil.copyfile('framework.cu', 'lc.cu')

# necessary functions
def update_enum(filename, comps, item):
  with open(filename, 'w') as f:
    f.write("#ifndef LC_" + item + "_H\n")
    f.write("#define LC_" + item + "_H\n\n")
    f.write("enum {NUL" + item)
    for c in comps:
      if item == 'VERIFIER':
        c = c[0:]
      else:
        c = c[2:]
      f.write(", " + str(c))

    f.write("};\n\n")


def update_gpu_components(filename, comps):
  with open(filename, 'a') as f:
    for c in comps:
      if c.startswith("v_"):
        c = c[2:]
      f.write("#include \"../" + str(c) + ".h\"\n")
    f.write("\n#endif\n")

# find components GPU
gfiles = next(os.walk('./components'))[2]
gpucomps = []
for f in gfiles:
  if f.startswith("d_"):
    gpucomps.append(f[:-2])


# sort components
gpucomps.sort()

# update constants
with open('./include/consts.h', 'w') as f:
  f.write("static const int CS = 1024 * 16;  // chunk size (in bytes) [do not change]\n")
  f.write("static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]\n")
  f.write("#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)\n")
  f.write("#define WS 64\n")
  f.write("#else\n")
  f.write("#define WS 32\n")
  f.write("#endif\n")

# update enum.h
update_enum('./components/include/GPUcomponents.h', gpucomps, 'GPUcomponents')


# update GPUcomponents.h
update_gpu_components('./components/include/GPUcomponents.h', gpucomps)

# find preprocessors GPU
gfiles = next(os.walk('./preprocessors'))[2]
gpupreprocess = []
for f in gfiles:
  if f.startswith("d_"):
    gpupreprocess.append(f[:-2])


#update preprocessor enum.h
update_enum('./preprocessors/include/GPUpreprocessors.h', gpupreprocess, 'GPUpreprocessors')


# update GPUpreprocessors.h
update_gpu_components('./preprocessors/include/GPUpreprocessors.h', gpupreprocess)

# find verifiers 
cfiles = next(os.walk('./verifiers'))[2]
gpuverifier = []
for f in cfiles:
  if f.endswith(".h"):
    gpuverifier.append("v_" + f[:-2])

# update enum.h
update_enum('./verifiers/include/verifiers.h', gpuverifier, 'VERIFIER')

# update verifiers.h
update_gpu_components('./verifiers/include/verifiers.h', gpuverifier)

file = 'lc.cu'
# update switch device encode
with open(file, "r+") as f:
  contents = f.read()
  m = re.search("##switch-device-encode-beg##[\s\S]*##switch-device-encode-end##", contents)
  str_to_add = ''
  for c in gpucomps:
    c = c[2:]
    str_to_add += "        case " + str(c) + ": good = d_" + str(c) + "(csize, in, out, temp); break;\n"
  contents = contents[:m.span()[0]] + "##switch-device-encode-beg##*/\n" + str_to_add + "        /*##switch-device-encode-end##" + contents[m.span()[1]:]
  f.seek(0)
  f.truncate()
  f.write(contents)

# update switch device decode
with open(file, "r+") as f:
  contents = f.read()
  m = re.search("##switch-device-decode-beg##[\s\S]*##switch-device-decode-end##", contents)
  str_to_add = ''
  for c in gpucomps:
    c = c[2:]
    str_to_add += "          case " + str(c) + ": d_i" + str(c) + "(csize, in, out, temp); break;\n"
  contents = contents[:m.span()[0]] + "##switch-device-decode-beg##*/\n" + str_to_add + "          /*##switch-device-decode-end##" + contents[m.span()[1]:]
  f.seek(0)
  f.truncate()
  f.write(contents)

# update switch pipeline
with open(file, "r+") as f:
  contents = f.read()
  m = re.search("##switch-pipeline-beg##[\s\S]*##switch-pipeline-end##", contents)
  str_to_add = ''
  for c in gpucomps:
    c = c[2:]
    str_to_add += "      case " + str(c) + ": s += \" " + str(c) + "\"; break;\n"
  contents = contents[:m.span()[0]] + "##switch-pipeline-beg##*/\n" + str_to_add + "      /*##switch-pipeline-end##" + contents[m.span()[1]:]
  f.seek(0)
  f.truncate()
  f.write(contents)

#update switch verify
with open(file, "r+") as f:
  contents = f.read()
  m = re.search("##switch-verify-beg##[\s\S]*##switch-verify-end##", contents)
  str_to_add = ''
  for c in gpuverifier:
    c = c[2:]
    str_to_add += "      case v_" + str(c) + ": " + str(c) + "(size, recon, orig, params.size(), params.data()); break;\n"
  contents = contents[:m.span()[0]] + "##switch-verify-beg##*/\n" + str_to_add + "      /*##switch-verify-end##" + contents[m.span()[1]:]
  f.seek(0)
  f.truncate()
  f.write(contents)


#update switch device preprocess encode
with open(file, "r+") as f:
  contents = f.read()
  m = re.search("##switch-device-preprocess-encode-beg##[\s\S]*##switch-device-preprocess-encode-end##", contents)
  str_to_add = ''
  for c in gpupreprocess:
    c = c[2:]
    str_to_add += "      case " + str(c) + ": d_" + str(c) + "(dpreencsize, dpreencdata, params.size(), params.data()); break;\n"
  contents = contents[:m.span()[0]] + "##switch-device-preprocess-encode-beg##*/\n" + str_to_add + "      /*##switch-device-preprocess-encode-end##" + contents[m.span()[1]:]
  f.seek(0)
  f.truncate()
  f.write(contents)

#update switch device preprocess decode
with open(file, "r+") as f:
  contents = f.read()
  m = re.search("##switch-device-preprocess-decode-beg##[\s\S]*##switch-device-preprocess-decode-end##", contents)
  str_to_add = ''
  for c in gpupreprocess:
    c = c[2:]
    str_to_add += "      case " + str(c) + ": d_i" + str(c) + "(dpredecsize, dpredecdata, params.size(), params.data()); break;\n"
  contents = contents[:m.span()[0]] + "##switch-device-preprocess-decode-beg##*/\n" + str_to_add + "      /*##switch-device-preprocess-decode-end##" + contents[m.span()[1]:]
  f.seek(0)
  f.truncate()
  f.write(contents)

# update enum map
with open(file, "r+") as f:
    contents = f.read()
    m = re.search("##component-map-beg##[\s\S]*##component-map-end##", contents)
    str_to_add = ''
    i = 0
    for c in gpucomps:
        c = c[2:]
        #print(c)
        i += 1
        str_to_add += "  components[\"" + str(c) + "\"] = " + str(i) + ";\n"
    contents = contents[:m.span()[0]] + "##component-map-beg##*/\n" + str_to_add + "  /*##component-map-end##" + contents[m.span()[1]:]
    f.seek(0)
    f.truncate()
    f.write(contents)

# update preprocessor map
with open(file, "r+") as f:
    contents = f.read()
    m = re.search("##preprocessor-map-beg##[\s\S]*##preprocessor-map-end##", contents)
    str_to_add = ''
    for c in gpupreprocess:
        c = c[2:]
        str_to_add += "  preprocessors[\"" + str(c) + "\"] = " + str(c) + ";\n"
    contents = contents[:m.span()[0]] + "##preprocessor-map-beg##*/\n" + str_to_add + "  /*##preprocessor-map-end##" + contents[m.span()[1]:]
    f.seek(0)
    f.truncate()
    f.write(contents)

# update verifier map
with open(file, "r+") as f:
    contents = f.read()
    m = re.search("##verifier-map-beg##[\s\S]*##verifier-map-end##", contents)
    str_to_add = ''
    for c in gpuverifier:
        c = c[2:]
        str_to_add += "  verifs[\"" + str(c) + "\"] = v_" + str(c) + ";\n"
    contents = contents[:m.span()[0]] + "##verifier-map-beg##*/\n" + str_to_add + "  /*##verifier-map-end##" + contents[m.span()[1]:]
    f.seek(0)
    f.truncate()
    f.write(contents)

# messages
print("Compile with\nnvcc -O3 -arch=sm_70 -DUSE_GPU -Xcompiler \"-O3 -march=native -fopenmp\" -o lc lc.cu\n")
print("Run the following command to see the usage message\n./lc")
