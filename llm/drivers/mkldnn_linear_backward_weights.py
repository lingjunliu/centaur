import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn import functional as F

    grad_output = torch.tensor(input_dict["grad_output"])
    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])

    if not cpu:
        grad_output = grad_output.cuda()
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()

    result = torch.matmul(grad_output.transpose(0, 1), input_tensor)

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
        grad_output = tf.constant(input_dict["grad_output"])
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])

        result = tf.matmul(tf.transpose(grad_output), input_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "grad_output": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "weight": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()