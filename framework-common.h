#ifndef LC_FRAMEWORK_COMMON_H
#define LC_FRAMEWORK_COMMON_H
#include <vector>
#include <string>
#include <map>
#include <cstdint>

using byte = unsigned char;
inline constexpr unsigned int max_stages = 8;
void verify(const int size, const std::byte* const recon, const std::byte* const orig, std::vector<std::pair<std::byte, std::vector<double>>> verifs);
std::map<std::string, byte> getVerifMap();
std::map<std::string, byte> getPreproMap();
std::map<std::string, byte> getCompMap();
std::string getPipeline(unsigned long long pipeline, const int stages);
void h_encode(const unsigned long long chain, const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int& outsize, uint32_t nthreads);
void h_decode(const unsigned long long chain, const byte* const __restrict__ input, byte* const __restrict__ output, int& outsize, uint32_t n_threads);
void h_preprocess_encode(int& hpreencsize, byte*& hpreencdata, std::vector<std::pair<byte, std::vector<double>>> prepros);
void h_preprocess_decode(int& hpredecsize, byte*& hpredecdata, std::vector<std::pair<byte, std::vector<double>>> prepros);
#endif /* end of include guard: LC_FRAMEWORK_COMMON_H */
