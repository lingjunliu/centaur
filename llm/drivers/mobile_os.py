import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale = input_dict.get("scale", 1.0)
    zero_point = input_dict.get("zero_point", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.quantize_per_tensor(input_tensor, scale, zero_point, torch.quint8)

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
        scale = input_dict.get("scale", 1.0)
        zero_point = input_dict.get("zero_point", 0)

        quantized_tensor = tf.quantization.fake_quant_with_min_max_vars(
            input_tensor,
            min=-zero_point * scale,
            max=(255 - zero_point) * scale,
            num_bits=8
        )
        
        quantized_tensor = tf.round((quantized_tensor / scale) + zero_point)
        quantized_tensor = tf.clip_by_value(quantized_tensor, 0, 255)
        result = tf.cast(quantized_tensor, tf.uint8).numpy()
        
    return {"result": result}

def main():
    A_TOL = 1.0

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "scale": 0.1,
        "zero_point": 10
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()