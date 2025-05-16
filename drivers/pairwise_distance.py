import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    x1_tensor = torch.tensor(input["x1"])
    x2_tensor = torch.tensor(input["x2"])
    p = input.get("p", 2.0)
    eps = input.get("eps", 1e-06)
    keepdim = input.get("keepdim", False)

    if not cpu:
        x1_tensor = x1_tensor.cuda()
        x2_tensor = x2_tensor.cuda()

    # Apply to torch.nn.functional.pairwise_distance
    distance = torch.nn.functional.pairwise_distance(x1_tensor, x2_tensor, p=p, eps=eps, keepdim=keepdim)

    if not cpu:
        distance = distance.cpu()

    return {"pairwise_distance": distance.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        x1_tensor = tf.constant(input["x1"])
        x2_tensor = tf.constant(input["x2"])
        p = input.get("p", 2.0)
        eps = input.get("eps", 1e-06)
        keepdim = input.get("keepdim", False)

        def pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False):
            diff = tf.abs(x1 - x2) if p == 1.0 else tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=-1) + eps)
            if keepdim:
                return tf.expand_dims(diff, -1)
            else:
                return diff

        # Apply the TensorFlow equivalent
        distance = pairwise_distance(x1_tensor, x2_tensor, p=p, eps=eps, keepdim=keepdim)

        return {"pairwise_distance": distance.numpy()}

def main():
    # Example input
    input_data = {
        "x1": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "x2": np.array([[0.4, 0.2, 0.6], [0.5, 0.7, 1.0]], dtype=np.float32),  # Ensure input is float type
        "p": 2.0,
        "eps": 1e-06,
        "keepdim": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_array = np.array(torch_result["pairwise_distance"])
    tf_array = np.array(tf_result["pairwise_distance"])

    if np.allclose(torch_array, tf_array):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()