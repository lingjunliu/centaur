import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    dim = input_dict.get("dim", 1)
    eps = input_dict.get("eps", 1e-08)

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()

    result = torch.nn.functional.cosine_similarity(input1, input2, dim=dim, eps=eps)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input1 = tf.constant(input_dict["input1"])
        input2 = tf.constant(input_dict["input2"])
        dim = input_dict.get("dim", 1)
        eps = input_dict.get("eps", 1e-08)
    
        norm_input1 = tf.norm(input1, axis=dim, keepdims=True)
        norm_input2 = tf.norm(input2, axis=dim, keepdims=True)
    
        numerator = tf.reduce_sum(tf.multiply(input1, input2), axis=dim, keepdims=True)
        denominator = tf.multiply(norm_input1, norm_input2)
    
        result = numerator / (denominator + eps)
        result = tf.squeeze(result, axis=dim)
    
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input1": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "input2": np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype=np.float32),
        "dim": 1,
        "eps": 1e-8
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()