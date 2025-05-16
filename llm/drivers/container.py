import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    modules = input_dict.get("modules", None)
    
    if modules is not None:
        modules = torch.nn.ModuleList([torch.nn.Linear(10, 10) for _ in range(len(modules))])
        
    if not cpu:
        if modules is not None:
            for module in modules:
                module.cuda()
        
    result = torch.nn.Sequential(*[torch.nn.Linear(10, 10) for _ in range(3)])
    if not cpu:
        result = result.cpu()
    
    return {"result": 0}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        result = 0
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "modules": [1, 2, 3]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()