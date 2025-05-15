import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    threshold = input_dict["threshold"]
    value = input_dict.get("value", 0.0)
    inplace = input_dict.get("inplace", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    threshold_module = torch.nn.Threshold(threshold, value, inplace=inplace)
    result = threshold_module(input_tensor)
    
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
        threshold_value = input_dict["threshold"]
        value = input_dict.get("value", 0.0)
        inplace = input_dict.get("inplace", False)

        result = tf.where(input_tensor > threshold_value, input_tensor, tf.constant(value, dtype=input_tensor.dtype))
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-0.5, 0, 0.5, 1], dtype=np.float32),
        "threshold": 0.2,
        "value": -1.0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()