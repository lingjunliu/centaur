import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.attention

    backend = input_dict.get("backend", "MATH")

    if backend == "ERROR":
        backend_enum = torch.nn.attention.SDPBackend.ERROR
    elif backend == "MATH":
        backend_enum = torch.nn.attention.SDPBackend.MATH
    elif backend == "FLASH_ATTENTION":
        backend_enum = torch.nn.attention.SDPBackend.FLASH_ATTENTION
    elif backend == "EFFICIENT_ATTENTION":
        backend_enum = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
    elif backend == "CUDNN_ATTENTION":
        backend_enum = torch.nn.attention.SDPBackend.CUDNN_ATTENTION
    else:
        raise ValueError(f"Invalid backend: {backend}")
    
    if not cpu:
        torch.cuda.init()
    
    return {"result": backend}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    backend = input_dict.get("backend", "MATH")
    
    return {"result": backend}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "backend": "FLASH_ATTENTION"
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()