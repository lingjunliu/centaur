import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])
    dtype = input_dict.get("dtype", torch.qint8)

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()

    q_weight = torch.quantize_per_tensor(input_tensor, float(scale.item()), int(zero_point.item()), dtype)

    if not cpu:
        q_weight = q_weight.cpu()

    return {"result": q_weight.int_repr().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = tf.constant(input_dict["scale"], dtype=tf.float32)
        zero_point = tf.constant(input_dict["zero_point"], dtype=tf.int32)
        dtype = input_dict.get("dtype", tf.qint8)
        
        min_range = tf.reduce_min(input_tensor)
        max_range = tf.reduce_max(input_tensor)
        
        quantized_tensor, min_range, max_range = tf.quantization.quantize_v2(input_tensor, min_range, max_range, T=dtype)

        result = quantized_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "scale": np.array([0.1], dtype=np.float32),
        "zero_point": np.array([0], dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()