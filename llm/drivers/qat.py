import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"].astype(np.float32))

    if not cpu:
        input_tensor = input_tensor.cuda()

    dequantize = torch.nn.quantized.DeQuantize()
    quantized_input = torch.quantize_per_tensor(input_tensor, scale=1.0, zero_point=0, dtype=torch.quint8)

    result = dequantize(quantized_input)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"].astype(np.float32))
    min_range = tf.cast(tf.reduce_min(input_dict["input"]), tf.float32)
    max_range = tf.cast(tf.reduce_max(input_dict["input"]), tf.float32)

    if not cpu:
        with tf.device("/GPU:0"):
            quantized_tensor, min_range, max_range = tf.quantization.quantize(input_tensor, min_range, max_range, T=tf.quint8)
            result = tf.quantization.dequantize(quantized_tensor, min_range, max_range, mode='MIN_COMBINED')
    else:
        with tf.device("/CPU:0"):
            quantized_tensor, min_range, max_range = tf.quantization.quantize(input_tensor, min_range, max_range, T=tf.quint8)
            result = tf.quantization.dequantize(quantized_tensor, min_range, max_range, mode='MIN_COMBINED')
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.int8)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()