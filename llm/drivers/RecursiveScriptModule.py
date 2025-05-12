import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self):
            return torch.tensor(1.0)

    module = MyModule()
    input_tensor = torch.jit.script(module)

    if not cpu:
        pass
    
    result = input_tensor()
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        result = tf.constant(1.0, dtype=tf.float32)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()