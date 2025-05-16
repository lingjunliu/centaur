import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    mat1 = torch.tensor(input_dict["mat1"])
    mat2 = torch.tensor(input_dict["mat2"])
    beta = input_dict.get("beta", 1.0)
    alpha = input_dict.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        mat1 = mat1.cuda()
        mat2 = mat2.cuda()

    result = beta * input_tensor + alpha * torch.matmul(mat1, mat2)

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
        input_tensor = tf.constant(input_dict["input"])
        mat1 = tf.constant(input_dict["mat1"])
        mat2 = tf.constant(input_dict["mat2"])
        beta = input_dict.get("beta", 1.0)
        alpha = input_dict.get("alpha", 1.0)

        result = beta * input_tensor + alpha * tf.matmul(mat1, mat2)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "mat1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "mat2": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
        "beta": 0.5,
        "alpha": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()