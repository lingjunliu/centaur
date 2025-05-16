import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"], dtype=torch.int64)
    quant_min = input_dict.get("quant_min", 0)
    quant_max = input_dict.get("quant_max", 255)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()
    
    result = torch.quantize_per_tensor(input_tensor, float(scale.item()), int(zero_point.item()), torch.quint8, quant_min=quant_min, quant_max=quant_max)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.int_repr().numpy(), "scale": np.array(result.q_scale()), "zero_point": np.array(result.q_zero_point())}

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
        quant_min = input_dict.get("quant_min", 0)
        quant_max = input_dict.get("quant_max", 255)

        scale = tf.cast(scale, dtype=tf.float32)
        zero_point = tf.cast(zero_point, dtype=tf.float32)

        quantized = tf.round(input_tensor / scale + zero_point)
        quantized = tf.clip_by_value(quantized, quant_min, quant_max)
        quantized = tf.cast(quantized, dtype=tf.uint8)

        result = quantized.numpy()
    
    return {"result": result, "scale": input_dict["scale"], "zero_point": input_dict["zero_point"]}

def main():
    A_TOL = 1

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "scale": np.array(0.1, dtype=np.float32),
        "zero_point": np.array(100, dtype=np.int32),
        "quant_min": 0,
        "quant_max": 255
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].astype(np.int32), tf_result["result"].astype(np.int32), atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["scale"], tf_result["scale"], atol=A_TOL), "Scale values do not match"
    assert np.allclose(torch_result["zero_point"], tf_result["zero_point"], atol=A_TOL), "Zero Point values do not match"

    print("Success")

if __name__ == "__main__":
    main()