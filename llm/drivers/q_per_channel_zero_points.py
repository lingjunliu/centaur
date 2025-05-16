import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])
    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        scale = scale.cuda()
        zero_point = zero_point.cuda()
        input_tensor = input_tensor.cuda()

    dtype = torch.quint8
    result = torch.quantize_per_channel(input_tensor, scale, zero_point, 0, dtype)
    result = result.int_repr()

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
        scale = tf.constant(input_dict["scale"], dtype=tf.float32)
        zero_point = tf.constant(input_dict["zero_point"], dtype=tf.float32)
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        
        quantized = tf.round(input_tensor / scale + zero_point)
        result = tf.cast(quantized, tf.int32)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 1.0

    input_data = {
        "scale": np.array([1.0, 2.0], dtype=np.float32),
        "zero_point": np.array([0, 2], dtype=np.float32),
        "input": np.array([1.0, 4.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "scale": np.array([1.0, 2.0, 0.5], dtype=np.float32),
        "zero_point": np.array([0, 2, 1], dtype=np.float32),
        "input": np.array([1.0, 4.0, 1.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()