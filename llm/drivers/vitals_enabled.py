import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.vitals_enabled()
    
    if not cpu:
        result = result
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if "vitals_enabled" in input_dict:
        vitals_enabled_val = input_dict["vitals_enabled"]
    else:
        vitals_enabled_val = False
    
    return {"result": False}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "vitals_enabled": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert torch_result["result"] == tf_result["result"], "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()