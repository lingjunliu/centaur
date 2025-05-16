import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.nn.Softsign
    m = torch.nn.Softsign()
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        m = m.cuda()

    output_tensor = m(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"softsign_output": output_tensor.numpy()}

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

        # Apply to TensorFlow equivalent (tf.math.softsign)
        output_tensor = tf.math.softsign(input_tensor)

        return {"softsign_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-0.5, 0.3, 0.8], [0.2, -0.6, 0.9]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result["softsign_output"]
    tf_output = tf_result["softsign_output"]

    if np.allclose(torch_output, tf_output, rtol=1e-05, atol=1e-08):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()