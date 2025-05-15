import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()
    
    result = torch.quantize_per_tensor(input_tensor, scale.item(), zero_point.item(), torch.quint8)
    
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
        scale = input_dict["scale"]
        zero_point = input_dict["zero_point"]

        quantized = tf.cast(tf.round(input_tensor / scale + zero_point), tf.uint8)

        result = quantized.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "scale": np.array(0.1, dtype=np.float32),
        "zero_point": np.array(0, dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()