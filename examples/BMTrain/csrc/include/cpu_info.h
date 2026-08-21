#include <cpuid.h>

static void cpuid(int info[4], int InfoType){
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}

int get_cpu_level() {
    //  SIMD: 128-bit
    bool HW_F16C;

    //  SIMD: 256-bit
    bool HW_AVX;
    bool HW_FMA;

    //  SIMD: 512-bit
    bool HW_AVX512F;    //  AVX512 Foundation

    int info[4];
    cpuid(info, 0);
    int nIds = info[0];

    //  Detect Features
    if (nIds >= 0x00000001){
        cpuid(info,0x00000001);
        HW_AVX    = (info[2] & ((int)1 << 28)) != 0;
        HW_FMA    = (info[2] & ((int)1 << 12)) != 0;
        HW_F16C   = (info[2] & ((int)1 << 29)) != 0;
    }
    if (nIds >= 0x00000007){
        cpuid(info,0x00000007);
        HW_AVX512F     = (info[1] & ((int)1 << 16)) != 0;
    }

    int ret = 0;
    if (HW_AVX && HW_FMA && HW_F16C) ret = 1;
    if (HW_AVX512F) ret = 2;
    return ret;
}
