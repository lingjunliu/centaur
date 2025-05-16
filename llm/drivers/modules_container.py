import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    modules_count = input_dict.get("modules", 0)
    modules = []
    for _ in range(modules_count):
        modules.append(torch.nn.Linear(10, 10))
    
    module_list = torch.nn.ModuleList(modules)

    if not cpu:
        for module in module_list:
            module.cuda()

    result = torch.nn.Sequential(*module_list)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": repr(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    modules_count = input_dict.get("modules", 0)
    modules = []
    for _ in range(modules_count):
        modules.append(tf.keras.layers.Dense(10))

    model = tf.keras.models.Sequential(modules)

    # Build the model with a dummy input
    model.build(input_shape=(None, 10))

    return {"result": repr(model)}

def main():

    input_data = {
        "modules": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_str = torch_result["result"]
    tf_str = tf_result["result"]

    def cleanup(s):
        s = s.replace(" ", "")
        s = s.replace("<", "")
        s = s.replace(">", "")
        s = s.replace("at0x", "at0")
        return s

    torch_str = cleanup(torch_str)
    tf_str = cleanup(tf_str)
    
    assert torch_str == tf_str

    print("Success")

if __name__ == "__main__":
    main()