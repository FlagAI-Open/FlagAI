#include <string>
#include <unordered_map>
#include "perf.cuh"

// 全局统一性能数据存储的实现
std::unordered_map<std::string, PerfData>& get_perf_data() {
    static std::unordered_map<std::string, PerfData> g_perf_data;
    return g_perf_data;
}
