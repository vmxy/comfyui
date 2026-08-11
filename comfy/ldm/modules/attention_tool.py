import torch
import logging


def is_fp8_supported():
    """
    通用检测当前硬件是否支持FP8计算
    支持: NVIDIA GPU, AMD GPU, 华为昇腾 NPU
    """
    # 1. 检测 NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        return _is_fp8_supported_nvidia()
    
    # 2. 检测 AMD GPU (ROCm)
    if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return _is_fp8_supported_amd()
    
    # 3. 检测华为昇腾 NPU
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return _is_fp8_supported_ascend()
    
    # 4. 其他未知硬件
    logging.warning("Unknown hardware platform, FP8 support cannot be determined")
    return False


def _is_fp8_supported_nvidia():
    """NVIDIA GPU FP8检测"""
    try:
        # 检查1: CUDA版本 >= 11.8
        cuda_version = torch.version.cuda
        if cuda_version:
            ver_parts = cuda_version.split('.')
            major = int(ver_parts[0])
            minor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
            if major < 11 or (major == 11 and minor < 8):
                logging.debug(f"CUDA version {cuda_version} < 11.8")
                return False
        
        # 检查2: 计算能力 >= 9.0 (Hopper/Blackwell)
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (9, 0):
            logging.debug(f"Compute Capability {major}.{minor} < 9.0")
            return False
        
        # 检查3: 实际测试FP8张量 (PyTorch 2.1+)
        try:
            test_tensor = torch.tensor([1.0], device='cuda', dtype=torch.float8_e4m3fn)
            _ = test_tensor + test_tensor
            logging.info(f"NVIDIA GPU FP8 supported (CC {major}.{minor})")
            return True
        except (ImportError, AttributeError, RuntimeError, TypeError):
            # PyTorch版本可能不支持float8 dtype，但硬件支持
            logging.info(f"NVIDIA GPU FP8 hardware supported (CC {major}.{minor})")
            return True
            
    except Exception as e:
        logging.warning(f"NVIDIA FP8 detection failed: {e}")
        return False


def _is_fp8_supported_amd():
    """AMD GPU FP8检测 (ROCm)"""
    try:
        # 检查ROCm版本 (FP8需要ROCm 5.4+)
        hip_version = torch.version.hip
        if hip_version:
            ver_parts = hip_version.split('.')
            major = int(ver_parts[0])
            minor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
            if major < 5 or (major == 5 and minor < 4):
                logging.debug(f"ROCm version {hip_version} < 5.4")
                return False
        
        # 检查GPU架构 (FP8需要 gfx9xx 或 gfx11xx+)
        # 通过设备名称或架构判断
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        fp8_archs = ['gfx90a', 'gfx94x', 'gfx1100', 'gfx1200', 'gfx1201']
        if any(arch in device_name.lower() for arch in fp8_archs):
            logging.info(f"AMD GPU FP8 supported ({device_name})")
            return True
        
        # 尝试实际使用FP8 (如果ROCm支持)
        try:
            test_tensor = torch.tensor([1.0], device='cuda', dtype=torch.float8_e4m3fn)
            _ = test_tensor + test_tensor
            logging.info(f"AMD GPU FP8 supported (ROCm {hip_version})")
            return True
        except:
            # 硬件可能支持但PyTorch版本不支持
            pass
        
        logging.debug(f"AMD GPU may not support FP8: {device_name}")
        return False
        
    except Exception as e:
        logging.warning(f"AMD FP8 detection failed: {e}")
        return False


def _is_fp8_supported_ascend():
    """华为昇腾 NPU FP8检测"""
    try:
        # 检查CANN版本 (FP8需要CANN 7.0+)
        if hasattr(torch, 'version') and hasattr(torch.version, 'cann'):
            cann_version = torch.version.cann
            if cann_version:
                ver_parts = cann_version.split('.')
                major = int(ver_parts[0])
                if major < 7:
                    logging.debug(f"CANN version {cann_version} < 7.0")
                    return False
        else:
            # 如果没有CANN版本信息，尝试其他方式
            pass
        
        # 检查昇腾设备型号 (FP8支持需要Ascend 910B+)
        # 通过设备名称判断
        if torch.npu.is_available():
            device_count = torch.npu.device_count()
            if device_count > 0:
                device_name = torch.npu.get_device_name(0)
                # Ascend 910B, 910B2, 910C 等支持FP8
                fp8_models = ['910b', '910b2', '910c', '910b3']
                if any(model in device_name.lower() for model in fp8_models):
                    logging.info(f"Ascend NPU FP8 supported ({device_name})")
                    return True
        
        # 尝试实际使用FP8 (如果torch_npu支持)
        try:
            test_tensor = torch.tensor([1.0], device='npu:0', dtype=torch.float8_e4m3fn)
            _ = test_tensor + test_tensor
            logging.info("Ascend NPU FP8 supported")
            return True
        except (ImportError, AttributeError, RuntimeError, TypeError):
            # CANN版本可能不支持float8 dtype，但硬件可能支持
            pass
        
        logging.debug("Ascend NPU FP8 support uncertain")
        return False
        
    except Exception as e:
        logging.warning(f"Ascend FP8 detection failed: {e}")
        return False


# 使用示例
def init_sage_attention():
    """根据FP8支持情况初始化sageattn"""
    global SAGE_ATTENTION3_IS_AVAILABLE
    
    SAGE_ATTENTION3_IS_AVAILABLE = False
    print("test ") 
    try:
        from sageattn3 import sageattn3_blackwell
        
        if is_fp8_supported():
            SAGE_ATTENTION3_IS_AVAILABLE = True
            if torch.cuda.is_available():
                print(f"SageAttention3 enabled on {torch.cuda.get_device_name(0)}")
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                print(f"SageAttention3 enabled on {torch.npu.get_device_name(0)}")
            else:
                print("SageAttention3 enabled on current hardware")
        else:
            print("SageAttention3 disabled: FP8 not supported")
            
    except ImportError:
        print("SageAttention3 not installed")
    except Exception as e:
        print(f"SageAttention3 initialization failed: {e}")


#if __name__ == "__main__":
    #init_sage_attention()
