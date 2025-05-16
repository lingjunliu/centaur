import numpy as np

def torch_acosh_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.acosh
    result_tensor = torch.acosh(input_tensor)

    if not cpu:
        result_tensor = result_tensor.cpu()
        
    return {"acosh_result": result_tensor.numpy()}

def tensorflow_acosh_version(input, cpu=True):
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
        result_tensor = tf.acosh(input_tensor)

        return {"acosh_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1.3192, 1.9915, 1.9674, 1.7151], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_acosh_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_acosh_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both results to numpy arrays and compare
    torch_result_array = np.array(torch_result["acosh_result"])
    tf_result_array = np.array(tf_result["acosh_result"])

    if np.allclose(torch_result_array, tf_result_array, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()