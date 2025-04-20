import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input1_tensor = torch.tensor(input["input1"])
    input2_tensor = torch.tensor(input["input2"])
    dim = input.get("dim", 1)
    eps = input.get("eps", 1e-8)

    # Apply to torch.nn.CosineSimilarity
    cosine_similarity = torch.nn.CosineSimilarity(dim=dim, eps=eps)
    result = cosine_similarity(input1_tensor, input2_tensor)

    if not cpu:
        result = result.cpu()

    return {"cosine_similarity": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input1_tensor = tf.constant(input["input1"])
        input2_tensor = tf.constant(input["input2"])
        axis = input.get("dim", 1)
        eps = input.get("eps", 1e-8)

        # Compute norms
        norm1 = tf.norm(input1_tensor, axis=axis)
        norm2 = tf.norm(input2_tensor, axis=axis)
        
        # Compute dot product
        dot_product = tf.reduce_sum(tf.multiply(input1_tensor, input2_tensor), axis=axis)
        
        # Compute cosine similarity
        cosine_similarity = dot_product / tf.maximum(norm1 * norm2, eps)

        return {"cosine_similarity": cosine_similarity.numpy()}

def main():
    # Example input
    input_data = {
        "input1": np.random.randn(100, 128).astype(np.float32),
        "input2": np.random.randn(100, 128).astype(np.float32),
        "dim": 1,
        "eps": 1e-6
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Check if the results are "equal"
    if np.allclose(torch_result["cosine_similarity"], tf_result["cosine_similarity"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()