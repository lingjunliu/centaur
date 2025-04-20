import numpy as np

# Function to set seed for reproducibility
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    tensor1 = torch.tensor(input["tensor1"])
    tensor2 = torch.tensor(input["tensor2"])
    value = input.get("value", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        tensor1 = tensor1.cuda()
        tensor2 = tensor2.cuda()

    # Perform addcdiv operation
    result = torch.addcdiv(input_tensor, tensor1, tensor2, value=value)

    if not cpu:
        result = result.cpu()

    return { "addcdiv": result.numpy() }

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

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

        # Perform addcdiv equivalent operation
        result = input_tensor + value * tf.divide(tensor1, tensor2)

        return { "addcdiv": result.numpy() }

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "tensor1": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "tensor2": np.array([[0.2, 0.5, 0.7], [0.1, 0.9, 1.2]], dtype=np.float32),
        "value": 1.0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Check if the results are equal
    assert np.allclose(torch_result['addcdiv'], tf_result['addcdiv']), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()