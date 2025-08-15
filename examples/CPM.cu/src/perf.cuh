#ifndef PERF_H
#define PERF_H

#include <string>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath> // For std::isnan

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#endif

// #define ENABLE_PERF
// 性能测量开关，可以通过编译时定义ENABLE_PERF来启用
#ifdef ENABLE_PERF
#define PERF_ENABLED 1
#else
#define PERF_ENABLED 0
#endif

struct PerfData {
    std::chrono::high_resolution_clock::time_point start_time;
    double total_time = 0.0;
    int count = 0;
    bool is_running = false;
    std::string type = "CPU"; // "CPU" 或 "CUDA"
    
#ifdef __CUDACC__
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool events_created = false;
    
    ~PerfData() {
        if (events_created) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }
    }
#endif
};

#if PERF_ENABLED

// 前向声明 - 在perf.cpp中实现
std::unordered_map<std::string, PerfData>& get_perf_data();

// 统一初始化性能测量系统
#define perf_init() \
    do { \
        auto& perf_data = get_perf_data(); \
        for (auto& pair : perf_data) { \
            auto& data = pair.second; \
            if (data.type == "CUDA") { \
                if (data.events_created) { \
                    cudaEventDestroy(data.start_event); \
                    cudaEventDestroy(data.stop_event); \
                } \
            } \
        } \
        perf_data.clear(); \
    } while(0)

// CPU性能测量开始
#define perf_startf(label) \
    do { \
        auto& data = get_perf_data()[#label]; \
        data.type = "CPU"; \
        if (!data.is_running) { \
            data.start_time = std::chrono::high_resolution_clock::now(); \
            data.is_running = true; \
        } \
    } while(0)

// CPU性能测量停止
#define perf_stopf(label) \
    do { \
        auto& data = get_perf_data()[#label]; \
        if (data.is_running && data.type == "CPU") { \
            auto end_time = std::chrono::high_resolution_clock::now(); \
            auto duration = std::chrono::duration<double, std::milli>(end_time - data.start_time).count(); \
            data.total_time += duration; \
            data.count++; \
            data.is_running = false; \
        } \
    } while(0)

#ifdef __CUDACC__
// CUDA性能测量开始
#define cuda_perf_startf(label) \
    do { \
        auto& data = get_perf_data()[#label]; \
        data.type = "CUDA"; \
        if (!data.events_created) { \
            cudaEventCreate(&data.start_event); \
            cudaEventCreate(&data.stop_event); \
            data.events_created = true; \
        } \
        if (!data.is_running) { \
            cudaEventRecord(data.start_event); \
            data.is_running = true; \
        } \
    } while(0)

// CUDA性能测量停止
#define cuda_perf_stopf(label) \
    do { \
        auto& data = get_perf_data()[#label]; \
        if (data.is_running && data.type == "CUDA" && data.events_created) { \
            cudaEventRecord(data.stop_event); \
            cudaEventSynchronize(data.stop_event); \
            float elapsed_time; \
            cudaEventElapsedTime(&elapsed_time, data.start_event, data.stop_event); \
            data.total_time += elapsed_time; \
            data.count++; \
            data.is_running = false; \
        } \
    } while(0)

// 获取GPU内存使用情况
#define cuda_get_memory_usage(free_mem, total_mem) \
    do { \
        size_t free_bytes, total_bytes; \
        cudaMemGetInfo(&free_bytes, &total_bytes); \
        free_mem = free_bytes; \
        total_mem = total_bytes; \
    } while(0)

// CUDA作用域自动计时器
#define cuda_perf_scope(label) \
    struct CudaPerfScope_##label { \
        CudaPerfScope_##label() { cuda_perf_startf(label); } \
        ~CudaPerfScope_##label() { cuda_perf_stopf(label); } \
    } cuda_perf_scope_##label

// 新增: 用于在指定流上进行CUDA性能测量的宏
#define cuda_perf_start_on_stream_f(label, stream_val) \
    do { \
        auto& data = get_perf_data()[#label]; \
        data.type = "CUDA"; \
        if (!data.events_created) { \
            cudaEventCreate(&data.start_event); \
            cudaEventCreate(&data.stop_event); \
            data.events_created = true; \
        } \
        if (!data.is_running) { \
            cudaEventRecord(data.start_event, stream_val); \
            data.is_running = true; \
        } \
    } while(0)

#define cuda_perf_stop_on_stream_f(label, stream_val) \
    do { \
        auto& data = get_perf_data()[#label]; \
        if (data.is_running && data.type == "CUDA" && data.events_created) { \
            cudaEventRecord(data.stop_event, stream_val); \
            cudaEventSynchronize(data.stop_event); \
            float elapsed_time; \
            cudaEventElapsedTime(&elapsed_time, data.start_event, data.stop_event); \
            data.total_time += elapsed_time; \
            data.count++; \
            data.is_running = false; \
        } \
    } while(0)

#define cuda_perf_scope_on_stream(label, stream_val) \
    struct CudaPerfScopeOnStream_##label { \
        cudaStream_t s_val; \
        CudaPerfScopeOnStream_##label(cudaStream_t stream_arg) : s_val(stream_arg) { cuda_perf_start_on_stream_f(label, s_val); } \
        ~CudaPerfScopeOnStream_##label() { cuda_perf_stop_on_stream_f(label, s_val); } \
    } cuda_perf_scope_on_stream_##label(stream_val)

#else // __CUDACC__ (即没有CUDA支持时)
// 当没有CUDA支持时，CUDA宏变成空操作
#define cuda_perf_startf(label) do {} while(0)
#define cuda_perf_stopf(label) do {} while(0)
#define cuda_get_memory_usage(free_mem, total_mem) do {} while(0)
#define cuda_perf_scope(label) do {} while(0)
#define cuda_perf_start_on_stream_f(label, stream_val) do {} while(0)
#define cuda_perf_stop_on_stream_f(label, stream_val) do {} while(0)
#define cuda_perf_scope_on_stream(label, stream_val) do {} while(0)

#endif // __CUDACC__

// 统一输出性能统计摘要
#define perf_summary() \
    do { \
        cudaDeviceSynchronize(); /* 在总结前同步所有CUDA操作 */ \
        auto& perf_data = get_perf_data(); \
        std::cout << "\n=== Performance Summary ===" << std::endl; \
        std::cout << std::left << std::setw(30) << "Label" \
                  << std::setw(8) << "Type" \
                  << std::setw(8) << "Count" \
                  << std::setw(15) << "Total(ms)" \
                  << std::setw(15) << "Average(ms)" << std::endl; \
        std::cout << std::string(76, '-') << std::endl; \
        \
        for (auto& pair : perf_data) { \
            const auto& name = pair.first; \
            auto& data = pair.second; \
            if (data.count > 0) { \
                double avg = (data.count > 0) ? (data.total_time / data.count) : 0.0; \
                std::cout << std::left << std::setw(30) << name \
                          << std::setw(8) << data.type \
                          << std::setw(8) << data.count \
                          << std::setw(15) << std::fixed << std::setprecision(3) << data.total_time \
                          << std::setw(15) << std::fixed << std::setprecision(3) << avg << std::endl; \
            } \
        } \
        \
        bool has_cuda = false; \
        for (const auto& pair : perf_data) { \
            if (pair.second.type == "CUDA") { \
                has_cuda = true; \
                break; \
            } \
        } \
        \
        if (has_cuda) { \
            size_t free_mem, total_mem; \
            cuda_get_memory_usage(free_mem, total_mem); \
            std::cout << std::string(76, '-') << std::endl; \
            std::cout << "GPU Memory: " << (total_mem - free_mem) / (1024*1024) << "MB used / " \
                      << total_mem / (1024*1024) << "MB total" << std::endl; \
        } \
        std::cout << "============================" << std::endl; \
    } while(0)

// 获取指定标签的总时间（毫秒）
#define perf_get_total_time(label) \
    (get_perf_data().count(#label) ? get_perf_data()[#label].total_time : 0.0)

// 获取指定标签的平均时间（毫秒）
#define perf_get_avg_time(label) \
    (get_perf_data().count(#label) && get_perf_data()[#label].count > 0 ? \
     get_perf_data()[#label].total_time / get_perf_data()[#label].count : 0.0)

// 获取指定标签的调用次数
#define perf_get_count(label) \
    (get_perf_data().count(#label) ? get_perf_data()[#label].count : 0)

// 重置指定标签的性能数据
#define perf_reset(label) \
    do { \
        auto& perf_data = get_perf_data(); \
        if (perf_data.count(#label)) { \
            auto& data = perf_data[#label]; \
            if (data.type == "CUDA" && data.events_created) { \
                cudaEventDestroy(data.start_event); \
                cudaEventDestroy(data.stop_event); \
            } \
            perf_data[#label] = PerfData{}; \
        } \
    } while(0)

// CPU作用域自动计时器（RAII风格）
#define perf_scope(label) \
    struct PerfScope_##label { \
        PerfScope_##label() { perf_startf(label); } \
        ~PerfScope_##label() { perf_stopf(label); } \
    } perf_scope_##label

#else // PERF_ENABLED (即性能测量被禁用时)

// 当性能测量被禁用时，所有宏都变成空操作
#define perf_init() do {} while(0)
#define perf_startf(label) do {} while(0)
#define perf_stopf(label) do {} while(0)
#define perf_summary() do {} while(0)
#define perf_get_total_time(label) 0.0
#define perf_get_avg_time(label) 0.0
#define perf_get_count(label) 0
#define perf_reset(label) do {} while(0)
#define perf_scope(label) do {} while(0)

// CUDA相关的空操作宏 (因为PERF_ENABLED为false，所有CUDA相关的宏也是空操作)
#define cuda_perf_startf(label) do {} while(0)
#define cuda_perf_stopf(label) do {} while(0)
#define cuda_get_memory_usage(free_mem, total_mem) do {} while(0)
#define cuda_perf_scope(label) do {} while(0)
#define cuda_perf_start_on_stream_f(label, stream_val) do {} while(0)
#define cuda_perf_stop_on_stream_f(label, stream_val) do {} while(0)
#define cuda_perf_scope_on_stream(label, stream_val) do {} while(0)

#endif // PERF_ENABLED

#endif // PERF_H


