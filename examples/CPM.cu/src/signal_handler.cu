#include "signal_handler.cuh"
#include <dlfcn.h>

// 保存原有信号处理器的全局变量
std::map<int, void(*)(int)> original_handlers;

void init_signal_handlers() {
    // 保存并设置信号处理器
    original_handlers[SIGSEGV] = signal(SIGSEGV, signal_handler);  // 段错误
    original_handlers[SIGABRT] = signal(SIGABRT, signal_handler);  // 异常终止
    original_handlers[SIGFPE] = signal(SIGFPE, signal_handler);    // 浮点异常
    original_handlers[SIGILL] = signal(SIGILL, signal_handler);    // 非法指令
#ifdef SIGBUS
    original_handlers[SIGBUS] = signal(SIGBUS, signal_handler);    // 总线错误 (某些系统可能没有)
#endif
    original_handlers[SIGTERM] = signal(SIGTERM, signal_handler);  // 终止信号
    original_handlers[SIGINT] = signal(SIGINT, signal_handler);    // 中断信号 (Ctrl+C)
    
    std::cout << "Signal handlers initialized for common exceptions" << std::endl;
}

// TODO 修复和python traceback的协作
void signal_handler(int sig) {
    const char* signal_name = "Unknown";
    
    switch (sig) {
        case SIGSEGV: signal_name = "SIGSEGV (Segmentation fault)"; break;
        case SIGABRT: signal_name = "SIGABRT (Abort)"; break;
        case SIGFPE:  signal_name = "SIGFPE (Floating point exception)"; break;
        case SIGILL:  signal_name = "SIGILL (Illegal instruction)"; break;
#ifdef SIGBUS
        case SIGBUS:  signal_name = "SIGBUS (Bus error)"; break;
#endif
        case SIGTERM: signal_name = "SIGTERM (Termination)"; break;
        case SIGINT:  signal_name = "SIGINT (Interrupt)"; break;
    }
    
    std::cerr << "\n=== SIGNAL CAUGHT ===" << std::endl;
    std::cerr << "Signal: " << signal_name << " (" << sig << ")" << std::endl;
    std::cerr << "Process ID: " << getpid() << std::endl;
    std::cerr << "====================" << std::endl;
    
    // 打印栈帧信息
    print_stack_trace(50);
    
    std::cerr << "\nProgram terminated due to signal " << sig << std::endl;
    std::cerr.flush();
    std::cout.flush();
    
    // 查找并调用原有的信号处理器
    auto it = original_handlers.find(sig);
    if (it != original_handlers.end() && it->second != SIG_DFL && it->second != SIG_IGN) {
        std::cerr << "Calling original signal handler..." << std::endl;
        it->second(sig);
    }

    // 恢复默认信号处理并重新发送信号
    std::cerr << "Restoring default handler..." << std::endl;
    signal(sig, SIG_DFL);
    raise(sig);
}

void print_stack_trace(int max_frames) {
    void **array = new void*[max_frames];
    
    // 获取调用栈
    int size = backtrace(array, max_frames);
    char **strings = backtrace_symbols(array, size);
    
    if (strings == nullptr) {
        std::cerr << "Failed to get backtrace symbols (backtrace may not be available on this system)" << std::endl;
        delete[] array;
        return;
    }
    
    // 添加基地址信息
    Dl_info dl_info;
    if (dladdr((void*)print_stack_trace, &dl_info)) {
        std::cerr << "\n=== MODULE INFO ===" << std::endl;
        std::cerr << "Base address: " << dl_info.dli_fbase << std::endl;
        std::cerr << "Module path: " << dl_info.dli_fname << std::endl;
    }
    
    std::cerr << "=== STACK TRACE ===" << std::endl;
    std::cerr << "Call stack (" << size << " frames):" << std::endl;
    
    for (int i = 0; i < size; i++) {
        std::string symbol_info = get_symbol_name(strings[i]);
        std::cerr << "[" << i << "] " << symbol_info << std::endl;
    }
    
    std::cerr << "==================" << std::endl;
    
    free(strings);
    delete[] array;
}

std::string get_symbol_name(const char* symbol) {
    std::string result(symbol);
    
    // 查找函数名的开始和结束位置
    char *start = strstr((char*)symbol, "(");
    char *end = strstr((char*)symbol, "+");
    
    if (start && end && start < end) {
        *end = '\0';
        char *function_name = start + 1;
        
        // 尝试demangle C++符号名
        int status;
        char *demangled = abi::__cxa_demangle(function_name, 0, 0, &status);
        
        if (status == 0 && demangled) {
            // 成功demangle
            std::string prefix(symbol, start - symbol + 1);
            std::string suffix = end + 1;
            result = prefix + demangled + "+" + suffix;
            free(demangled);
        } else {
            // demangle失败，恢复原始字符串
            *end = '+';
        }
    }
    
    return result;
} 