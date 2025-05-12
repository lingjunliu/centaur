import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale = float(input_dict["scale"])
    zero_point = int(input_dict.get("zero_point", 0))
    quant_min = input_dict.get("quant_min", 0)
    quant_max = input_dict.get("quant_max", 255)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.quantize_per_tensor(
        input_tensor,
        scale=scale,
        zero_point=zero_point,
        quant_min=quant_min,
        quant_max=quant_max,
        dtype=torch.quint8
    )

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = tf.constant(input_dict["scale"], dtype=tf.float32)
        zero_point = tf.constant(input_dict.get("zero_point", 0), dtype=tf.int32)
        quant_min = tf.constant(input_dict.get("quant_min", 0), dtype=tf.int32)
        quant_max = tf.constant(input_dict.get("quant_max", 255), dtype=tf.int32)

        input_tensor_scaled = input_tensor / scale + tf.cast(zero_point, dtype=tf.float32)
        input_tensor_clamped = tf.clip_by_value(input_tensor_scaled, tf.cast(quant_min, dtype=tf.float32), tf.cast(quant_max, dtype=tf.float32))
        result = tf.round(input_tensor_clamped)
        result = tf.cast(result, dtype=tf.uint8)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "scale": 0.1,
        "zero_point": 10,
        "quant_min": 0,
        "quant_max": 255
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()