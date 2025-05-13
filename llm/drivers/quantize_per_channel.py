import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])
    axis = input_dict.get("axis", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()

    result = torch.quantize_per_channel(input_tensor, scale, zero_point, axis, torch.qint8)

    if not cpu:
        result = result.cpu()

    return {"result": result.int_repr().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = tf.constant(input_dict["scale"], dtype=tf.float32)
        zero_point = tf.constant(input_dict["zero_point"], dtype=tf.int32)
        axis = input_dict.get("axis", 0)
        quant_min = input_dict.get("quant_min", 0)
        quant_max = input_dict.get("quant_max", 255)

        
        input_shape = input_tensor.shape
        scale_shape = scale.shape
        
        if len(input_shape) != len(scale_shape):
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[axis] = scale_shape[0]
            scale = tf.reshape(scale, broadcast_shape)
            zero_point = tf.reshape(zero_point, broadcast_shape)

        quantized = tf.clip_by_value(tf.round(input_tensor / scale + tf.cast(zero_point, dtype=tf.float32)), quant_min, quant_max)
        result = tf.cast(quantized, dtype=tf.int8)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "scale": np.array([0.1, 0.2], dtype=np.float32),
        "zero_point": np.array([10, 20], dtype=np.int32),
        "axis": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()