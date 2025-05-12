import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])
    quant_min = input_dict.get("quant_min", 0)
    quant_max = input_dict.get("quant_max", 255)

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()

    result = torch.fake_quantize_per_tensor_affine(input_tensor, scale, zero_point, quant_min=quant_min, quant_max=quant_max)

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
        input_tensor = tf.constant(input_dict["input"])
        scale = tf.constant(input_dict["scale"])
        zero_point = tf.cast(tf.constant(input_dict["zero_point"]), dtype=tf.float32)
        quant_min = input_dict.get("quant_min", 0)
        quant_max = input_dict.get("quant_max", 255)

        min_val = scale * (quant_min - zero_point)
        max_val = scale * (quant_max - zero_point)

        result = tf.clip_by_value(input_tensor, min_val, max_val)
        result = tf.round((result / scale) + zero_point)
        result = tf.clip_by_value(result, quant_min, quant_max)
        result = (result - zero_point) * scale
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "scale": np.array(0.5, dtype=np.float32),
        "zero_point": np.array(0, dtype=np.int64),
        "quant_min": 0,
        "quant_max": 255
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()