import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Apply Sigmoid activation using torch
    input_tensor = torch.tensor(input["input"])
    sigmoid = torch.nn.Sigmoid()
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        sigmoid = sigmoid.cuda()
    
    output_tensor = sigmoid(input_tensor)
    
    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"sigmoid_output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Apply Sigmoid activation using TensorFlow
        input_tensor = tf.constant(input["input"])
        output_tensor = tf.nn.sigmoid(input_tensor)

        return {"sigmoid_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).astype(np.float32)  # Generate random sample input
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Check if outputs are close enough (with a small tolerance for floating-point differences)
    if np.allclose(torch_result["sigmoid_output"], tf_result["sigmoid_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()