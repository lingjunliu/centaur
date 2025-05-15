import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding")

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = F.pad(input_tensor, (padding[1], padding[1], padding[0], padding[0]), mode='replicate')

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict.get("padding")

    if not cpu:
        device_string = "/GPU:0"
    else:
        device_string = "/CPU:0"

    with tf.device(device_string):
        paddings = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
        
        input_shape = input_tensor.shape
        
        if input_shape[1] < padding[0] or input_shape[2] < padding[1]:
          paddings = [[0,0],[0,0],[0,0],[0,0]]

        result = tf.pad(input_tensor, paddings, mode="REFLECT")

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 1, 5, 5).astype(np.float32),
        "padding": (1, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()