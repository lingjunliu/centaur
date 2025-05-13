import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale = float(input_dict["scale"][0])
    zero_point = int(input_dict["zero_point"][0])
    quant_min = input_dict.get("quant_min", 0)
    quant_max = input_dict.get("quant_max", 255)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.quantize_per_tensor(input_tensor, scale, zero_point, torch.quint8)

    if not cpu:
        result = result.cpu()

    return {"result": result.int_repr().numpy(), "scale": np.array([result.q_scale()]), "zero_point": np.array([result.q_zero_point()])}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = input_dict["scale"][0]
        zero_point = input_dict["zero_point"][0]
        quant_min = input_dict.get("quant_min", 0)
        quant_max = input_dict.get("quant_max", 255)

        quantized = tf.clip_by_value(tf.round(input_tensor / scale + zero_point), quant_min, quant_max)
        result = tf.cast(quantized, tf.uint8)

        result = result.numpy()
        result_scale = scale
        result_zero_point = zero_point

    return {"result": result, "scale": np.array([result_scale]), "zero_point": np.array([result_zero_point])}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "scale": np.array([0.1]),
        "zero_point": np.array([100]),
        "quant_min": 0,
        "quant_max": 255
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(np.array([torch_result["scale"]]), tf_result["scale"], atol=A_TOL), "Scale do not match"
    assert np.allclose(np.array([torch_result["zero_point"]]), tf_result["zero_point"], atol=A_TOL), "Zero Point do not match"

    print("Success")

if __name__ == "__main__":
    main()