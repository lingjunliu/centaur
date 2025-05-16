import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    torch.set_autocast_cpu_enabled(input_dict['enabled'])
    
    return {"result": torch.is_autocast_cpu_enabled()}

def tensorflow_version(input_dict, cpu=True):
    
    return {"result": input_dict['enabled']}

def main():
    A_TOL = 0.01

    input_data = {
        "enabled": True
    }

    torch_result = torch_version(input_data)
    
    tf_result = tensorflow_version(input_data)
    
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    input_data = {
        "enabled": False
    }

    torch_result = torch_version(input_data)
    
    tf_result = tensorflow_version(input_data)
    
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()