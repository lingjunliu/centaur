import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    approximate = input.get("approximate", 'none')
    
    # Use torch GELU
    gelu = torch.nn.GELU(approximate=approximate)
    output = gelu(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"gelu_output": output.numpy()}

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
        approximate = input.get("approximate", 'none')

        # TensorFlow GELU
        if approximate == "tanh":
            sqrt_2_over_pi = np.sqrt(2 / np.pi)
            gelu_output = 0.5 * input_tensor * (1 + tf.nn.tanh(sqrt_2_over_pi * (input_tensor + 0.044715 * tf.pow(input_tensor, 3))))
        else:
            # Accurate computation for Gelu
            gelu_output = 0.5 * input_tensor * (1 + tf.math.erf(input_tensor / np.sqrt(2.0)))

        return {"gelu_output": gelu_output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).astype(np.float32),
        "approximate": 'none'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.allclose(torch_result["gelu_output"], tf_result["gelu_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()