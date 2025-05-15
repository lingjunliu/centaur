import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.AlphaDropout(p=p)
    m.train(training)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        alpha = -np.sqrt((1 - p) / (p + 1e-8))
        if training:
            dropout = tf.nn.dropout(input_tensor, rate=p)
            ret = alpha * tf.where(tf.equal(dropout, 0), tf.ones_like(dropout) * tf.reduce_mean(input_tensor), tf.zeros_like(dropout)) + dropout
        else:
            ret = input_tensor
        result = ret.numpy()

    return {"result": result}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "p": 0.5,
        "training": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "p": 0.5,
        "training": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()