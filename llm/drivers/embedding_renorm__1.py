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
    
    result = input_tensor

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        maxnorm = input_dict.get("maxnorm", 3.0)
        norm_type = input_dict.get("norm_type", 2.0)

        def renorm(v):
            l2_norm = tf.norm(v, ord=norm_type)
            condition = tf.greater(l2_norm, maxnorm)
            factor = tf.cond(condition, lambda: maxnorm / l2_norm, lambda: tf.constant(1.0, dtype=v.dtype))
            return v * factor
        
        result = tf.vectorized_map(renorm, input_tensor)

        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32),
        "maxnorm": 1.0,
        "norm_type": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()