import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    eps = input.get("eps", None)

    # Apply torch.logit
    if eps is not None:
        logit_tensor = torch.logit(input_tensor, eps=eps)
    else:
        logit_tensor = torch.logit(input_tensor)

    if not cpu:
        logit_tensor = logit_tensor.cpu()

    return {"logit": logit_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        eps = input.get("eps", None)

        # Apply TensorFlow equivalent
        if eps is not None:
            input_tensor = tf.clip_by_value(input_tensor, eps, 1 - eps)
        logit_tensor = tf.math.log(input_tensor / (1 - input_tensor))

        return {"logit": logit_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.1, 0.5, 0.8], dtype=np.float32),
        "eps": 1e-7
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_logit = np.array(torch_result["logit"])
    tf_logit = np.array(tf_result["logit"])

    # Use np.allclose to account for small floating-point differences
    if np.allclose(torch_logit, tf_logit, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()