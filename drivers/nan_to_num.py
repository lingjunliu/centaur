import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    nan = input_dict.get("nan", 0.0)
    posinf = input_dict.get("posinf", 0.0)
    neginf = input_dict.get("neginf", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nan_to_num(input_tensor, nan=nan, posinf=posinf, neginf=neginf)

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
        nan = input_dict.get("nan", 0.0)
        posinf = input_dict.get("posinf", 0.0)
        neginf = input_dict.get("neginf", 0.0)

        nan_mask = tf.math.is_nan(input_tensor)
        inf_mask = tf.math.is_inf(input_tensor)
        posinf_mask = tf.logical_and(inf_mask, input_tensor > 0)
        neginf_mask = tf.logical_and(inf_mask, input_tensor < 0)
        
        result = tf.where(nan_mask, tf.cast(nan, input_tensor.dtype), input_tensor)
        result = tf.where(posinf_mask, tf.cast(posinf, input_tensor.dtype), result)
        result = tf.where(neginf_mask, tf.cast(neginf, input_tensor.dtype), result)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([np.nan, np.inf, -np.inf, 1.0, 2.0], dtype=np.float32),
        "nan": 0.0,
        "posinf": 1.0,
        "neginf": -1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([np.nan, np.inf, -np.inf, 1.0, 2.0], dtype=np.float64),
        "nan": 5.0,
        "posinf": 10.0,
        "neginf": -10.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([np.nan, np.inf, -np.inf, 1.0, 2.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()