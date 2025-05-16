import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"])
    dim = input_dict.get("dim", 1)
    eps = input_dict.get("eps", 1e-8)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    result = torch.cosine_similarity(input_tensor, other_tensor, dim=dim, eps=eps)
    
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
        input_tensor = tf.constant(input_dict["input"])
        other_tensor = tf.constant(input_dict["other"])
        dim = input_dict.get("dim", 1)
        eps = input_dict.get("eps", 1e-8)

        input_norm = tf.linalg.norm(input_tensor, axis=dim, keepdims=True)
        other_norm = tf.linalg.norm(other_tensor, axis=dim, keepdims=True)

        input_normalized = input_tensor / (input_norm + eps)
        other_normalized = other_tensor / (other_norm + eps)

        result = tf.reduce_sum(input_normalized * other_normalized, axis=dim)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "other": np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "other": np.array([7.0, 8.0, 9.0], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()