import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input_dict["input"])
    diagonal = input_dict.get("diagonal", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.triu
    result = torch.triu(input_tensor, diagonal=diagonal)

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
        # Unpack input dictionary
        input_tensor = tf.constant(input_dict["input"])
        diagonal = input_dict.get("diagonal", 0)

        # Apply TensorFlow equivalent
        result = tf.experimental.numpy.triu(input_tensor.numpy(), k=diagonal)

        return {"result": result}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(3, 3).astype(np.float32),
        "diagonal": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if results are equal
    assert np.allclose(torch_result["result"], tf_result["result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()