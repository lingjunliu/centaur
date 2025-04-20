import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    p = input["p"]

    # Apply to torch.mvlgamma
    result_tensor = torch.special.multigammaln(input_tensor, p)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"mvlgamma_result": result_tensor.numpy()}

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
        p = input["p"]
        
        # TensorFlow equivalent using the gamma function
        def multi_gamma_log(x, d):
            d = tf.cast(d, tf.float32)
            x = tf.cast(x, tf.float32)
            log_pi = tf.constant(np.log(np.pi))
            terms = tf.map_fn(lambda i: tf.math.lgamma(x + 0.5 * (1 - i)), tf.range(1, d + 1, dtype=x.dtype), dtype=x.dtype)
            return (d * (d - 1) * log_pi) / 4.0 + tf.reduce_sum(terms, axis=0)
        
        result_tensor = multi_gamma_log(input_tensor, p)

    return {"mvlgamma_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "p": 3
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using numpy
    if np.allclose(torch_result["mvlgamma_result"], tf_result["mvlgamma_result"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()