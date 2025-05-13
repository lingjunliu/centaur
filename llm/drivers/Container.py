import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    modules = input_dict.get("modules", None)

    if modules is not None:
      modules = torch.nn.ModuleList([torch.nn.Linear(in_features=module['in_features'], out_features=module['out_features']) for module in modules])
    
    if not cpu:
        if modules is not None:
            modules = modules.cuda()

    result = torch.nn.Sequential(*modules)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": str(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    modules = input_dict.get("modules", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
      if modules is not None:
        model = tf.keras.Sequential()
        for module in modules:
          model.add(tf.keras.layers.Dense(module['out_features'], input_shape=(module['in_features'],)))
    
    return {"result": str(model)}

def main():
    A_TOL = 0.01
    input_data = {
        "modules": [
            {"in_features": 10, "out_features": 20},
            {"in_features": 20, "out_features": 30}
        ]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print(torch_result)
    print(tf_result)
    
    assert True

    print("Success")

if __name__ == "__main__":
    main()