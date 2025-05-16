import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Convert input to torch tensor
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.floor function
    result_tensor = torch.floor(input_tensor)
    
    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"floor_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert input to tensorflow tensor
        input_tensor = tf.constant(input["input"])
        
        # Apply tf.math.floor function
        result_tensor = tf.math.floor(input_tensor)

        return {"floor_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(4).astype(np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results ensuring reproducibility
    np.testing.assert_allclose(torch_result["floor_result"], tf_result["floor_result"], rtol=1e-5, atol=1e-8)
    print("equal")

if __name__ == "__main__":
    main()