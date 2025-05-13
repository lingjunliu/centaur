import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim0 = input_dict["dim0"]
    dim1 = input_dict["dim1"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.transpose(input_tensor, dim0, dim1)

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
        dim0 = input_dict["dim0"]
        dim1 = input_dict["dim1"]

        rank = len(input_tensor.shape)
        perm = list(range(rank))
        perm[dim0] = dim1
        perm[dim1] = dim0

        result = tf.transpose(input_tensor, perm=perm)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "dim0": 0,
        "dim1": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()