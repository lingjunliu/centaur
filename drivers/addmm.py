import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    mat1 = torch.tensor(input["mat1"])
    mat2 = torch.tensor(input["mat2"])
    beta = input.get("beta", 1)
    alpha = input.get("alpha", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        mat1 = mat1.cuda()
        mat2 = mat2.cuda()

    # Apply to torch.addmm
    result = torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)

    if not cpu:
        result = result.cpu()

    return {"addmm_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        mat1 = tf.constant(input["mat1"])
        mat2 = tf.constant(input["mat2"])
        beta = input.get("beta", 1)
        alpha = input.get("alpha", 1)

        # Apply to TensorFlow equivalent
        result = tf.add(beta * input_tensor, alpha * tf.linalg.matmul(mat1, mat2))

        result = result.numpy()

        return {"addmm_result": result}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "mat1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "mat2": np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=np.float32),
        "beta": 1,
        "alpha": 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare the results
    assert np.allclose(torch_result["addmm_result"], tf_result["addmm_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()
