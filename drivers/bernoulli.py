import numpy as np

# Function to set seed for reproducibility
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply torch.bernoulli
    result_tensor = torch.bernoulli(input_tensor)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"bernoulli": float(result_tensor.float().mean().item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        logits = tf.random.uniform(tf.shape(input_tensor))
        result_tensor = tf.cast(tf.less_equal(logits, input_tensor), tf.float32)

        return {"bernoulli": float(tf.reduce_mean(result_tensor).numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Since we capture the mean to avoid randomness affecting the comparison
    def compare_results(result1, result2):
        return np.isclose(result1, result2, atol=1e-6)

    if compare_results(torch_result["bernoulli"], tf_result["bernoulli"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()