import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    other = torch.tensor(input_dict["other"]) if isinstance(input_dict["other"], np.ndarray) else input_dict["other"]
    rounding_mode = input_dict.get("rounding_mode", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(other, torch.Tensor):
            other = other.cuda()

    result = torch.divide(input_tensor, other, rounding_mode=rounding_mode)

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
        other = tf.constant(input_dict["other"]) if isinstance(input_dict["other"], np.ndarray) else input_dict["other"]
        rounding_mode = input_dict.get("rounding_mode", None)

        result = tf.divide(input_tensor, other)

        if rounding_mode == 'floor':
          result = tf.floor(result)
        elif rounding_mode == 'trunc':
          result = tf.math.floor(result) if tf.reduce_any(result < 0) else tf.math.ceil(result)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([2.0, 4.0, 6.0], dtype=np.float32),
        "other": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([5, 6, 7], dtype=np.float32),
        "other": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([5.0, 6.0, 7.0], dtype=np.float32),
        "other": 2,
        "rounding_mode": "floor"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([-5.0, 6.0, -7.0], dtype=np.float32),
        "other": 2,
        "rounding_mode": "trunc"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    # tensorflow truncates towards zero, regardless of sign. We adjust for that in our function
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()