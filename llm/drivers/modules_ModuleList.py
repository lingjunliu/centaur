import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    modules = input_dict["modules"]

    module_list = torch.nn.ModuleList(modules)

    return {"result": len(module_list)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    modules = input_dict["modules"]

    return {"result": len(modules)}

def main():
    A_TOL = 0.01

    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_data = {
        "modules": [nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()