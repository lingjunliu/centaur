import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    enabled = input_dict.get("enabled", True)

    if not cpu:
        torch.cuda.set_device(0)

    torch.set_autocast_cache_enabled(enabled)
    
    result = torch.is_autocast_cache_enabled()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    enabled = input_dict.get("enabled", True)

    class AutocastCacheManager:
        _is_enabled = True
        
        @classmethod
        def set_enabled(cls, enabled):
            cls._is_enabled = enabled

        @classmethod
        def is_enabled(cls):
            return cls._is_enabled

    AutocastCacheManager.set_enabled(enabled)
    
    result = AutocastCacheManager.is_enabled()
    
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "enabled": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "enabled": True
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()