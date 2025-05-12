import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    mat = torch.tensor(input_dict["mat"])
    vec = torch.tensor(input_dict["vec"])
    beta = input_dict.get("beta", 1.0)
    alpha = input_dict.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        mat = mat.cuda()
        vec = vec.cuda()

    result = torch.addmv_(input_tensor, mat, vec, beta=beta, alpha=alpha)

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
        input_tensor = tf.Variable(input_dict["input"], dtype=tf.float32)
        mat = tf.constant(input_dict["mat"], dtype=tf.float32)
        vec = tf.constant(input_dict["vec"], dtype=tf.float32)
        beta = input_dict.get("beta", 1.0)
        alpha = input_dict.get("alpha", 1.0)

        mv = tf.linalg.matmul(mat, tf.expand_dims(vec, axis=1))
        mv = tf.squeeze(mv, axis=1)

        result = input_tensor.assign(beta * input_tensor + alpha * mv)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "mat": np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]], dtype=np.float32),
        "vec": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "beta": 0.5,
        "alpha": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()