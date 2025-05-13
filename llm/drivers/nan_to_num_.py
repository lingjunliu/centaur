import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    nan = input_dict.get("nan", 0.0)
    posinf = input_dict.get("posinf", None)
    neginf = input_dict.get("neginf", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if posinf is not None:
            posinf = torch.tensor(posinf).cuda()
        if neginf is not None:
            neginf = torch.tensor(neginf).cuda()

    result = torch.nan_to_num(input_tensor, nan=nan, posinf=posinf, neginf=neginf)

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
        nan = input_dict.get("nan", 0.0)
        posinf = input_dict.get("posinf", None)
        neginf = input_dict.get("neginf", None)

        input_tensor = tf.where(tf.math.is_nan(input_tensor), nan, input_tensor)

        if posinf is not None:
            input_tensor = tf.where(tf.math.is_inf(input_tensor) & (input_tensor > 0), posinf, input_tensor)

        if neginf is not None:
            input_tensor = tf.where(tf.math.is_inf(input_tensor) & (input_tensor < 0), neginf, input_tensor)

        result = input_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([np.nan, np.inf, -np.inf, 1.0, 2.0], dtype=np.float32),
        "nan": 0.0,
        "posinf": 10.0,
        "neginf": -10.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()