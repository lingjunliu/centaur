import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Compute logarithm to the base 10
    result_tensor = torch.log10(input_tensor)

    # If required, move the result to CPU
    if not cpu:
        result_tensor = result_tensor.cuda()
    else:
        result_tensor = result_tensor.cpu()

    return {"log10_result": result_tensor.cpu().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Compute logarithm to the base 10
        result_tensor = tf.math.log(input_tensor) / tf.math.log(tf.constant(10, dtype=input_tensor.dtype))

        return {"log10_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32)
    }

    # Mark device preference
    use_cpu = True

    # Torch example
    torch_result = torch_version(input_data, cpu=use_cpu)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=use_cpu)
    print("TensorFlow result:", tf_result)

    # Compare results
    assert np.allclose(torch_result["log10_result"], tf_result["log10_result"], atol=1e-6), "Results are not equal"
    print("Results are equal") # This will only print if the assertion passes

if __name__ == "__main__":
    main()