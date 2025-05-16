import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    beta = input_dict.get("beta", 1.0)
    threshold = input_dict.get("threshold", 20.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.Softplus(beta=beta, threshold=threshold)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        beta = input_dict.get("beta", 1.0)
        threshold = input_dict.get("threshold", 20.0)

        result = tf.math.log(tf.math.exp(tf.clip_by_value(input_tensor, clip_value_min=-threshold, clip_value_max=threshold) * beta) + 1) / beta

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "beta": 1.0,
        "threshold": 20.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()