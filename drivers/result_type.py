import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    tensor1 = torch.tensor(input["tensor1"])
    tensor2 = torch.tensor(input["tensor2"])

    # Apply to torch.result_type
    result_type = torch.result_type(tensor1, tensor2)

    # Return result type as the string representation of its name
    return {"result_type": str(result_type).split('.')[-1]}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensor1 = tf.constant(input["tensor1"])
        tensor2 = tf.constant(input["tensor2"])

        # Apply the equivalent TensorFlow operation
        result_dtype = tf.experimental.numpy.result_type(tensor1, tensor2)  # Equivalent to numpy's result_type but in TensorFlow

        # Convert the dtype to a standard string representation (ensure it matches PyTorch's string representation)
        result_type_str = result_dtype.name if result_dtype.name != 'float16' else 'float32'

        return {"result_type": result_type_str}

def main():
    # Example input
    input_data = {
        "tensor1": np.array([0.5, 0.3, 0.8], dtype=np.float32),
        "tensor2": np.array([0.2, 0.6, 0.9], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if torch_result["result_type"] == tf_result["result_type"]:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()