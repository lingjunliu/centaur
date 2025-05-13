import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    try:
        result = torch.view_as_complex(input_tensor)
    except RuntimeError as e:
        if "Tensor must have a last dimension of size 2" in str(e):
            input_tensor = input_tensor.reshape(*input_tensor.shape[:-1], -1, 2)
            result = torch.view_as_complex(input_tensor)
        else:
            raise e

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
        input_tensor = tf.constant(input_dict["input"])
        
        input_shape = input_tensor.shape
        
        if len(input_shape) == 0:
            raise ValueError("Input tensor must have at least one dimension.")
        
        if input_shape[-1] % 2 != 0:
            
            input_tensor = tf.reshape(input_tensor, (*input_shape[:-1], -1, 2))
            input_shape = input_tensor.shape

        if input_shape[-1] != 2:
            input_tensor = tf.reshape(input_tensor, (*input_shape[:-1], -1, 2))
            input_shape = input_tensor.shape
        
        real = input_tensor[..., 0]
        imag = input_tensor[..., 1]

        result = tf.complex(real, imag)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()