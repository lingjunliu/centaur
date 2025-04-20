import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    lower = input.get("lower", 1./8)
    upper = input.get("upper", 1./3)
    training = input.get("training", False)

    # Apply to torch.nn.functional.rrelu_
    if not cpu:
        input_tensor = input_tensor.cuda()

    result_tensor = torch.nn.functional.rrelu_(
        input_tensor, lower=lower, upper=upper, training=training
    )

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"rrelu_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        lower = input.get("lower", 1./8)
        upper = input.get("upper", 1./3)
        training = input.get("training", False)

        if training:
            # Generate random alpha from uniform distribution
            alpha = tf.random.uniform(
                shape=tf.shape(input_tensor),
                minval=lower,
                maxval=upper,
                dtype=tf.float32
            )
        else:
            alpha = (lower + upper) / 2

        # Apply RReLU activation
        result_tensor = tf.where(input_tensor >= 0, input_tensor, alpha * input_tensor)

        return {"rrelu_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(2, 3).astype(np.float32),
        "lower": 1./8,
        "upper": 1./3,
        "training": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert results in a common format
    torch_output = np.array(torch_result["rrelu_result"], dtype=np.float32)
    tf_output = np.array(tf_result["rrelu_result"], dtype=np.float32)

    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()