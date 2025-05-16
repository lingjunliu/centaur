import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    min_val = input_dict.get("min", None)
    max_val = input_dict.get("max", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if min_val is not None:
            min_val = torch.tensor(min_val).cuda()
        if max_val is not None:
            max_val = torch.tensor(max_val).cuda()

    if min_val is None and max_val is None:
        return {"result": input_tensor.numpy()}
    
    input_tensor.clamp_(min=min_val, max=max_val)
    
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
        min_val = input_dict.get("min", None)
        max_val = input_dict.get("max", None)

        if min_val is not None:
            min_val = tf.constant(min_val, dtype=input_tensor.dtype)
            input_tensor = tf.maximum(input_tensor, min_val)

        if max_val is not None:
            max_val = tf.constant(max_val, dtype=input_tensor.dtype)
            input_tensor = tf.minimum(input_tensor, max_val)

        result = input_tensor.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "min": 0.0,
        "max": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "min": -0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "max": 1.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()