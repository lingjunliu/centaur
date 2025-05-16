import numpy as np

# Function to set seed for reproducibility
def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Use deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(input["mode"])

    # For this example, we can create a simple calculation to demonstrate determinism
    data = torch.tensor(input["data"])
    
    if not cpu:
        data = data.cuda()

    result = torch.add(data, 1.0)  # Simple operation to demonstrate determinism

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
        tf.config.experimental.enable_op_determinism()
    else:
        device_string = "/gpu:0"
        tf.config.experimental.enable_op_determinism()

    with tf.device(device_string):
        # For this example, we can create a simple calculation to demonstrate determinism
        data = tf.constant(input["data"])
        result = tf.add(data, 1.0)  # Simple operation to demonstrate determinism

        return {"result": result.numpy()}

def main():
    # Example input
    input_data = {
        "mode": True,
        "data": np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.9], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert to check if results are equal
    assert np.array_equal(torch_result["result"], tf_result["result"]), "Results are not equal."
    
    print("equal")

if __name__ == "__main__":
    main()