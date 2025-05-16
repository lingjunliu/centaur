import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    modules = input_dict.get("modules", None)
    if modules is not None:
        modules = torch.nn.ModuleList([torch.nn.Linear(in_features=10, out_features=5) for _ in range(len(modules))])

    if not cpu:
        if modules is not None:
            for module in modules:
                module.cuda()

    result = torch.nn.Sequential(*modules) if modules is not None else torch.nn.Sequential()

    if not cpu:
        result = result.cpu()

    return {"result": None}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    modules = input_dict.get("modules", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if modules is not None:
            tf_modules = []
            for _ in range(len(modules)):
                tf_modules.append(tf.keras.layers.Dense(units=5, activation=None, input_shape=(10,)))
            result = tf.keras.Sequential(tf_modules)
        else:
            result = tf.keras.Sequential()

    return {"result": None}

def main():
    A_TOL = 0.01
    input_data = {
        "modules": [1, 2, 3]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    print("Success")

if __name__ == "__main__":
    main()