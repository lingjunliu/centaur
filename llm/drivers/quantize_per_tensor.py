import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict.get("scale", 1.0))
    zero_point = torch.tensor(input_dict.get("zero_point", 0))
    dtype = input_dict.get("dtype", torch.quint8)

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()
    
    result = torch.quantize_per_tensor(input_tensor, scale.item(), int(zero_point.item()), dtype)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.dequantize().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = tf.constant(input_dict.get("scale", 1.0), dtype=tf.float32)
        zero_point = tf.constant(input_dict.get("zero_point", 0), dtype=tf.int32)

        quantized_tensor = tf.quantization.quantize(input_tensor,
                                                     input_tensor.numpy().min(),
                                                     input_tensor.numpy().max(),
                                                     tf.quint8)

        dequantized_tensor = tf.quantization.dequantize(quantized_tensor[0],
                                                         input_tensor.numpy().min(),
                                                         input_tensor.numpy().max())
    
        result = dequantized_tensor.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "scale": 0.1,
        "zero_point": 10,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()