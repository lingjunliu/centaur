import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    qconfig = input_dict.get("qconfig", torch.quantization.default_qconfig)
    reduce_range = input_dict.get("reduce_range", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.quantize_per_tensor_dynamic(input_tensor, reduce_range=reduce_range, dtype=torch.qint8)
    
    if not cpu:
        result = result.cpu()
    
    scale = result.q_scale()
    zero_point = result.q_zero_point()
    
    dequantized = scale * (result.int_repr().float() - zero_point)
    
    return {"result": dequantized.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        
        min_range = tf.reduce_min(input_tensor)
        max_range = tf.reduce_max(input_tensor)
        
        scale = (max_range - min_range) / 255.0 if (max_range - min_range) != 0 else 1.0
        zero_point = -min_range / scale if scale != 0 else 0.0
        
        zero_point = tf.clip_by_value(tf.round(zero_point), 0, 255)
        
        quantized = tf.round(input_tensor / scale + zero_point)
        quantized = tf.clip_by_value(quantized, 0, 255)
        quantized = tf.cast(quantized, tf.int8)

        dequantized = (tf.cast(quantized, tf.float32) - zero_point) * scale
        
        result = dequantized.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.1
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "reduce_range": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()