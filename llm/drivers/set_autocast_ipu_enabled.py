import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    torch.set_autocast_ipu_enabled(input_dict.get("enabled", True))
    
    return {"result": np.array(torch.is_autocast_ipu_enabled())}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    enabled = input_dict.get("enabled", True)

    class IPUAutocastContext:
        def __init__(self, enabled):
            self.enabled = enabled

        def __enter__(self):
            global _ipu_autocast_enabled
            self.original_state = _ipu_autocast_enabled
            _ipu_autocast_enabled = self.enabled
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            global _ipu_autocast_enabled
            _ipu_autocast_enabled = self.original_state
    
    global _ipu_autocast_enabled
    _ipu_autocast_enabled = enabled
    
    return {"result": np.array(_ipu_autocast_enabled)}

def main():
    A_TOL = 0.01

    input_data = {
        "enabled": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "enabled": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    _ipu_autocast_enabled = True
    main()