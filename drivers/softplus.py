from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    beta = input.get("beta", 1)
    threshold = input.get("threshold", 20)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.softplus
    output = torch.nn.functional.softplus(input_tensor, beta=beta, threshold=threshold)

    if not cpu:
        output = output.cpu()

    return {"softplus_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        beta = input.get("beta", 1)
        threshold = input.get("threshold", 20)

        # Apply the equivalent custom Softplus implementation
        def custom_softplus(x, beta, threshold):
            return tf.where(
                x * beta > threshold,
                x,
                (1 / beta) * tf.math.log(1 + tf.math.exp(beta * x))
            )

        output = custom_softplus(input_tensor, beta, threshold)

        return {"softplus_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "beta": 1,
        "threshold": 20
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    assert np.allclose(torch_result["softplus_output"], tf_result["softplus_output"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()