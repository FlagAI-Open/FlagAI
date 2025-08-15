#pragma once

#include <signal.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <string>
#include <iostream>
#include <map>

// 保存原有信号处理器
extern std::map<int, void(*)(int)> original_handlers;

// 初始化signal处理器
void init_signal_handlers();

// signal处理函数
void signal_handler(int sig);

// 打印栈帧信息
void print_stack_trace(int max_frames = 50);

// 获取符号名称（带demangling）
std::string get_symbol_name(const char* symbol); 