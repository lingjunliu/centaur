import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    device = input_tensor.device
    device_str = "cpu" if cpu else "cuda"

    if not cpu:
        input_tensor = input_tensor.cuda()
        device = input_tensor.device
        device_str = "cuda"

    try:
        result = torch.get_device_module(device).__name__
    except RuntimeError:
        result = device_str

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])

    device_str = "cpu" if cpu else "cuda"

    if cpu:
        device = "/cpu:0"
    else:
        device = "/gpu:0"

    with tf.device(device):
        try:
            tf_device = tf.raw_ops.XlaGetDeviceModule(tensor=input_tensor)
            result = tf_device.numpy().decode('utf-8')
            if result == "":
              result = device_str
        except Exception as e:
            result = device_str

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"].lower() == tf_result["result"].lower(), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)

    assert torch_result["result"].lower() == tf_result["result"].lower(), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()