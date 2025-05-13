import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    true_val = input_dict.get("true_val", 1.0)
    false_val = input_dict.get("false_val", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.where(input_tensor > 0, torch.tensor(true_val), torch.tensor(false_val))

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
        true_val = tf.constant(input_dict.get("true_val", 1.0))
        false_val = tf.constant(input_dict.get("false_val", 0.0))

        result = tf.where(input_tensor > 0, true_val, false_val)

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32),
        "true_val": 1.0,
        "false_val": 0.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32),
        "true_val": 5.0,
        "false_val": -5.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()