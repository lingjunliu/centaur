import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    tensor1 = torch.tensor(input["tensor1"])
    tensor2 = torch.tensor(input["tensor2"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        tensor1 = tensor1.cuda()
        tensor2 = tensor2.cuda()

    value = input.get("value", 1.0)

    # Apply torch.addcmul
    output_tensor = torch.addcmul(input_tensor, tensor1, tensor2, value=value)

    # Ensure output is on CPU for comparison
    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"addcmul_output": output_tensor.numpy()}

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
        tensor1 = tf.constant(input["tensor1"])
        tensor2 = tf.constant(input["tensor2"])
        value = input.get("value", 1.0)

        # Apply TensorFlow equivalent operations
        output_tensor = tf.add(input_tensor, value * tf.multiply(tensor1, tensor2))

    return {"addcmul_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "tensor1": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        "tensor2": np.array([[0.2, 0.6, 0.9], [0.7, 0.8, 0.2]], dtype=np.float32),
        "value": 0.1,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print whether the results are equal or not
    torch_output = torch_result["addcmul_output"]
    tf_output = tf_result["addcmul_output"]

    # Use np.allclose to consider floating-point precision issues
    if np.allclose(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()