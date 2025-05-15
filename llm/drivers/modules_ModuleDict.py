import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    modules = input_dict["modules"]
    torch_modules = {}
    for key, val in modules.items():
        torch_modules[key] = torch.nn.Linear(val.shape[1], val.shape[0]) if len(val.shape) > 1 else torch.nn.Linear(1, 1)
        with torch.no_grad():
            torch_modules[key].weight = nn.Parameter(torch.tensor(val))
            torch_modules[key].bias = nn.Parameter(torch.zeros(val.shape[0] if len(val.shape) > 1 else 1))

    if not cpu:
        for key in torch_modules:
            torch_modules[key] = torch_modules[key].cuda()
    
    moduledict = torch.nn.ModuleDict(torch_modules)

    if not cpu:
        result = {key: moduledict[key].weight.cpu().detach().numpy() for key in moduledict}
    else:
        result = {key: moduledict[key].weight.detach().numpy() for key in moduledict}

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    modules = input_dict["modules"]
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        tf_modules = {}
        for key, val in modules.items():
            initializer = tf.constant_initializer(val)
            tf_modules[key] = tf.compat.v1.get_variable(key, shape=val.shape, initializer=initializer, trainable=False)
            
        result = {key: tf_modules[key].numpy() for key in tf_modules}

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "modules": {
            "linear": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "conv": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        }
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for key in torch_result["result"]:
        assert np.allclose(torch_result["result"][key], tf_result["result"][key], atol=A_TOL), f"Results do not match for key {key}"

    print("Success")

if __name__ == "__main__":
    main()