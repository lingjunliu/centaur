import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply torch.nn.LogSigmoid
    m = torch.nn.LogSigmoid()
    output_tensor = m(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"log_sigmoid_output": output_tensor.numpy().tolist()}

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

        # Apply TensorFlow equivalent
        output_tensor = tf.math.log(1 / (1 + tf.exp(-input_tensor)))

        return {"log_sigmoid_output": output_tensor.numpy().tolist()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [-0.2, 0.6, -0.9]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare outputs
    if np.allclose(torch_result["log_sigmoid_output"], tf_result["log_sigmoid_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()