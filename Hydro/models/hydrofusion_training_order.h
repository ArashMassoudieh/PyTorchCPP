#pragma once
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace hydrofusion {
// Permute only already-assembled TRAIN windows. Local RNG leaves model RNG alone.
inline std::vector<int64_t> seededForecastWindowOrder(int64_t count,int seed,int epoch){
 if(count<1||seed<0||epoch<1)throw std::invalid_argument("Invalid seeded forecast TRAIN order");
 std::vector<int64_t> v(static_cast<size_t>(count));std::iota(v.begin(),v.end(),0);
 std::seed_seq key{static_cast<uint32_t>(seed),0u,static_cast<uint32_t>(epoch),0u,0x48594653u};
 std::mt19937_64 rng(key);std::shuffle(v.begin(),v.end(),rng);return v;
}
}
