import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.nn.Softmax2d
    softmax2d = torch.nn.Softmax2d()
    if not cpu:
        input_tensor = input_tensor.cuda()
        softmax2d = softmax2d.cuda()

    output = softmax2d(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"softmax2d_output": output.numpy()}

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

        # Apply to TensorFlow equivalent
        output = tf.nn.softmax(input_tensor, axis=-3) # TensorFlow does not have exact equivalent, using axis=-3 for channels

        return {"softmax2d_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3, 12, 13).astype(np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print whether they are equal
    np.testing.assert_almost_equal(torch_result["softmax2d_output"], tf_result["softmax2d_output"], decimal=5, err_msg="The results are not equal!")
    print("equal")

if __name__ == "__main__":
    main()