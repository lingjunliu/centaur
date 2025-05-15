import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    x1 = torch.tensor(input_dict["x1"])
    x2 = torch.tensor(input_dict["x2"])
    p = input_dict.get("p", 2.0)
    eps = input_dict.get("eps", 1e-8)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        x1 = x1.cuda()
        x2 = x2.cuda()

    pairwise_distance = torch.nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
    result = pairwise_distance(x1, x2)

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
        x1 = tf.constant(input_dict["x1"])
        x2 = tf.constant(input_dict["x2"])
        p = input_dict.get("p", 2.0)
        eps = input_dict.get("eps", 1e-8)
        keepdim = input_dict.get("keepdim", False)

        diff = tf.abs(x1 - x2)
        result = tf.reduce_sum(tf.pow(diff, p), axis=-1, keepdims=keepdim)
        result = tf.pow(result + eps, 1/p)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "x1": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "x2": np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32),
        "p": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "x1": np.array([[1, 2, 3]], dtype=np.float32),
        "x2": np.array([[7, 8, 9]], dtype=np.float32),
        "p": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "x1": np.array([[1, 2, 3]], dtype=np.float32),
        "x2": np.array([[7, 8, 9]], dtype=np.float32),
        "p": 1.0,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()