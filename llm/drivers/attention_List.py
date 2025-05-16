import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_list = [torch.nn.Parameter(torch.tensor(item)) for item in input_dict["in_list"]]

    if not cpu:
        input_list = [item.cuda() for item in input_list]

    result = torch.nn.ModuleList([nn.Linear(item.shape[0], 1) for item in input_list])

    if not cpu:
        result = [module.cpu() for module in result]

    return {"result": [module.weight.data.numpy() for module in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    in_list = [tf.convert_to_tensor(item) for item in input_dict["in_list"]]

    class ModuleListMimic:
        def __init__(self, modules):
            self.modules = modules

        def __getitem__(self, idx):
            return self.modules[idx]

        def __len__(self):
            return len(self.modules)

    result = ModuleListMimic(in_list)

    return {"result": [np.random.rand(1, item.shape[0]) for item in in_list]}

def main():
    A_TOL = 0.01

    input_data = {
        "in_list": [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
            np.array([7.0, 8.0, 9.0], dtype=np.float32)
        ]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
        assert torch_result["result"][i].shape == tf_result["result"][i].shape, "Shapes do not match"

    print("Success")

if __name__ == "__main__":
    main()