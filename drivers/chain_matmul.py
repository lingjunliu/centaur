import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary and convert lists to tensors
    if cpu:
        matrices = [torch.tensor(matrix) for matrix in input["matrices"]]
    else:
        matrices = [torch.tensor(matrix).cuda() for matrix in input["matrices"]]

    # Perform chained matrix multiplication
    result = torch.chain_matmul(*matrices)

    if not cpu:
        result = result.cpu()

    return {"output": result.detach().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary and convert lists to tensors
        matrices = [tf.constant(matrix) for matrix in input["matrices"]]

        # Perform chained matrix multiplication by reducing tf.matmul
        result = matrices[0]
        for matrix in matrices[1:]:
            result = tf.matmul(result, matrix)

        return {"output": result.numpy()}

def main():
    # Example input
    input_data = {
        "matrices": [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
            np.array([[9, 10], [11, 12]], dtype=np.float32)
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality and print the result
    if np.array_equal(torch_result["output"], tf_result["output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()