import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    scale = input_dict.get("scale", 1.0)
    zero_point = input_dict.get("zero_point", 0)
    dtype = input_dict.get("dtype", torch.qint8)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=dtype)
    result = result.dequantize()

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
        scale = input_dict.get("scale", 1.0)
        zero_point = input_dict.get("zero_point", 0)
        dtype = input_dict.get("dtype", tf.qint8)

        quantized = tf.divide(input_tensor, scale)
        quantized = tf.round(quantized)
        quantized = tf.add(quantized, zero_point)

        if dtype == tf.qint8:
            min_val = -128
            max_val = 127
        elif dtype == tf.quint8:
            min_val = 0
            max_val = 255
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        quantized = tf.clip_by_value(quantized, min_val, max_val)
        
        quantized = tf.cast(quantized, dtype)
        
        quantized = tf.cast(quantized, tf.float32)
        quantized = tf.subtract(quantized, zero_point)
        quantized = tf.multiply(quantized, scale)
        
        result = quantized.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "scale": 0.1,
        "zero_point": 0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()