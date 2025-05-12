import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    maxnorm = input_dict.get("maxnorm", 3.0)
    norm_type = input_dict.get("norm_type", 2.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    for i in range(input_tensor.shape[0]):
        norm = torch.linalg.norm(input_tensor[i], ord=norm_type)
        if norm > maxnorm:
            input_tensor[i] = input_tensor[i] * (maxnorm / norm)

    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_np = input_dict["input"]
    input_tensor = tf.constant(input_np)
    maxnorm = input_dict.get("maxnorm", 3.0)
    norm_type = input_dict.get("norm_type", 2.0)
    
    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):
        l2_norms = tf.norm(input_tensor, ord=norm_type, axis=-1, keepdims=True)
        
        scale = maxnorm / l2_norms
        scale = tf.minimum(scale, 1.0)

        renormed_tensor = input_tensor * scale
    
    return {"result": renormed_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "maxnorm": 2.0,
        "norm_type": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()