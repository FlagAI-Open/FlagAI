#!/usr/bin/env python3
"""
Flash Attention 版本检查和兼容性测试脚本
"""

import sys
import os

def check_flash_attn_import():
    """检查 Flash Attention 是否可以导入"""
    try:
        import flash_attn
        print(f"✓ Flash Attention 已安装")
        print(f"  版本: {flash_attn.__version__}")
        return True, flash_attn.__version__
    except ImportError as e:
        print(f"✗ Flash Attention 未安装: {e}")
        return False, None

def check_flash_attn_modules():
    """检查 Flash Attention 主要模块"""
    modules_to_check = [
        ('flash_attn.bert_padding', ['unpad_input', 'pad_input', 'index_first_axis']),
        ('flash_attn.flash_attn_interface', ['flash_attn_unpadded_qkvpacked_func', 'flash_attn_func', 'flash_attn_varlen_kvpacked_func']),
        ('flash_attn.ops.rms_norm', ['RMSNorm']),
        ('flash_attn.layers.rotary', ['RotaryEmbedding']),
    ]
    
    results = []
    for module_name, attributes in modules_to_check:
        try:
            module = __import__(module_name, fromlist=attributes)
            missing_attrs = []
            for attr in attributes:
                if not hasattr(module, attr):
                    missing_attrs.append(attr)
            if missing_attrs:
                print(f"✗ {module_name}: 缺少属性 {missing_attrs}")
                results.append(False)
            else:
                print(f"✓ {module_name}: 所有属性可用")
                results.append(True)
        except ImportError as e:
            print(f"✗ {module_name}: 导入失败 - {e}")
            results.append(False)
    
    return all(results)

def check_cuda_compatibility():
    """检查 CUDA 兼容性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠ CUDA 不可用（Flash Attention 需要 CUDA）")
            return False
    except ImportError:
        print("✗ PyTorch 未安装")
        return False

def check_pytorch_version():
    """检查 PyTorch 版本"""
    try:
        import torch
        version = torch.__version__
        print(f"✓ PyTorch 版本: {version}")
        # Flash Attention 3.0 需要 PyTorch 1.12+
        major, minor = map(int, version.split('.')[:2])
        if major > 1 or (major == 1 and minor >= 12):
            print("  PyTorch 版本满足要求 (>=1.12)")
            return True
        else:
            print("  ⚠ PyTorch 版本可能过低 (需要 >=1.12)")
            return False
    except ImportError:
        print("✗ PyTorch 未安装")
        return False

def test_flash_attn_functionality():
    """测试 Flash Attention 功能"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠ 跳过功能测试（CUDA 不可用）")
            return None
        
        # 尝试导入不同的 API
        flash_attn_func = None
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            try:
                from flash_attn.flash_attn_interface import flash_attn_func
            except ImportError:
                print("⚠ 无法导入 flash_attn_func，跳过功能测试")
                return None
        
        if flash_attn_func is None:
            print("⚠ 无法导入 flash_attn_func，跳过功能测试")
            return None
        
        # 创建测试数据
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        
        # 测试 flash_attn_func
        output = flash_attn_func(q, k, v, causal=True)
        print(f"✓ Flash Attention 功能测试通过")
        print(f"  输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Flash Attention 功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Flash Attention 版本检查和兼容性测试")
    print("=" * 60)
    print()
    
    results = []
    
    # 检查导入
    has_flash_attn, version = check_flash_attn_import()
    results.append(("Flash Attention 安装", has_flash_attn))
    
    if has_flash_attn:
        # 检查版本
        if version:
            try:
                from packaging import version as pkg_version
                v = pkg_version.parse(version)
                if v >= pkg_version.parse("3.0.0"):
                    print(f"  ✓ 版本 >= 3.0.0 (推荐)")
                elif v >= pkg_version.parse("2.0.0"):
                    print(f"  ✓ 版本 >= 2.0.0 (可用)")
                else:
                    print(f"  ⚠ 版本 < 2.0.0 (建议升级)")
            except:
                pass
        
        # 检查模块
        results.append(("Flash Attention 模块", check_flash_attn_modules()))
    
    # 检查 PyTorch
    results.append(("PyTorch 版本", check_pytorch_version()))
    
    # 检查 CUDA
    results.append(("CUDA 兼容性", check_cuda_compatibility()))
    
    # 测试功能（如果 CUDA 可用）
    try:
        import torch
        if torch.cuda.is_available():
            func_result = test_flash_attn_functionality()
            if func_result is not None:
                results.append(("Flash Attention 功能", func_result))
    except:
        pass
    
    # 打印结果
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    # 总结
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败，请检查上述错误")
        return 1

if __name__ == "__main__":
    sys.exit(main())

