import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])
    dtype = input_dict.get("dtype", torch.quint8)

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()
        
    q_tensor = torch.quantize_per_tensor(input_tensor, scale.item(), zero_point.item(), dtype)

    if not cpu:
        q_tensor = q_tensor.cpu()
    
    return {"result": q_tensor.dequantize().numpy()}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = tf.constant(input_dict["scale"], dtype=tf.float32)
        zero_point = tf.constant(input_dict["zero_point"], dtype=tf.float32)
        
        q_min = tf.cast(tf.reduce_min(input_tensor), tf.float32)
        q_max = tf.cast(tf.reduce_max(input_tensor), tf.float32)

        quantized, min_range, max_range = tf.quantization.quantize(input_tensor, q_min, q_max, tf.quint8)
        
        dequantized = tf.quantization.dequantize(quantized, q_min, q_max)
        
        result = dequantized.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.1
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "scale": np.array([0.5], dtype=np.float32),
        "zero_point": np.array([0], dtype=np.int32),
        "dtype": torch.quint8
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()