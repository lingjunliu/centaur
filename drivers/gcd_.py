import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    other = torch.tensor(input_dict["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other = other.cuda()

    torch.gcd_(input_tensor, other)

    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        other = tf.constant(input_dict["other"])

        input_np = input_tensor.numpy()
        other_np = other.numpy()

        if isinstance(input_np, np.ndarray) and isinstance(other_np, np.ndarray):
            result = np.gcd(input_np, other_np)
        elif isinstance(input_np, np.ndarray):
             result = np.gcd(input_np, other_np)
        elif isinstance(other_np, np.ndarray):
            result = np.gcd(input_np, other_np)
        else:
            result = np.gcd(input_np, other_np)
            
        input_tensor = result
    return {"result": input_tensor}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([12, 18, 24], dtype=np.int32),
        "other": np.array([6, 9, 12], dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([12], dtype=np.int32),
        "other": np.array([6], dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([12], dtype=np.int32),
        "other": np.array(6, dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()