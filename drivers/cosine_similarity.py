import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    x1_tensor = torch.tensor(input["x1"])
    x2_tensor = torch.tensor(input["x2"])
    dim = input.get("dim", 1)
    eps = input.get("eps", 1e-8)

    if not cpu:
        x1_tensor = x1_tensor.cuda()
        x2_tensor = x2_tensor.cuda()

    # Apply to torch.nn.functional.cosine_similarity
    similarity = torch.nn.functional.cosine_similarity(x1_tensor, x2_tensor, dim=dim, eps=eps)

    if not cpu:
        similarity = similarity.cpu()

    return {"cosine_similarity": similarity.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        x1_tensor = tf.constant(input["x1"])
        x2_tensor = tf.constant(input["x2"])
        dim = input.get("dim", 1)
        eps = input.get("eps", 1e-8)

        # Apply to TensorFlow equivalent
        normalize_a = tf.nn.l2_normalize(x1_tensor, dim, epsilon=eps)
        normalize_b = tf.nn.l2_normalize(x2_tensor, dim, epsilon=eps)
        similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=dim)

        return {"cosine_similarity": similarity.numpy()}

def main():
    # Example input
    input_data = {
        "x1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "x2": np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "dim": 1,
        "eps": 1e-8
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_similarity = torch_result["cosine_similarity"]
    tf_similarity = tf_result["cosine_similarity"]

    if np.allclose(torch_similarity, tf_similarity, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()