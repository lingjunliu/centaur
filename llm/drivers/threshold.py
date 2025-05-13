import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    threshold = torch.tensor(input_dict["threshold"])
    value = input_dict.get("value", 0.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        threshold = threshold.cuda()
    
    result = torch.threshold(input_tensor, threshold, value)
    
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
        threshold = tf.constant(input_dict["threshold"])
        value = input_dict.get("value", 0.0)

        result = tf.where(input_tensor > threshold, input_tensor, value)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([-0.2, 0.1, 0.5, 1.2], dtype=np.float32),
        "threshold": np.array(0.3, dtype=np.float32),
        "value": 0.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()