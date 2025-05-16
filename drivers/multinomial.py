import numpy as np

def torch_multinomial_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input["input"])
    num_samples = input["num_samples"]
    replacement = input.get("replacement", False)
    generator = input.get("generator", None)
    
    if not cpu:
        input_tensor = input_tensor.to('cuda')

    sampled_indices = torch.multinomial(
        input_tensor, num_samples, replacement, generator=generator, out=None
    )

    if not cpu:
        sampled_indices = sampled_indices.cpu()

    return {"sampled_indices": sampled_indices.numpy()}

def tensorflow_multinomial_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        num_samples = input["num_samples"]
        replacement = input.get("replacement", False)

        def sample_without_replacement(input_tensor, num_samples):
            indices = tf.argsort(input_tensor, direction='DESCENDING')
            selected_indices = tf.slice(indices, [0], [num_samples])
            return selected_indices

        if not replacement:
            sampled_indices = sample_without_replacement(input_tensor, num_samples)
        else:
            sampled_indices = tf.random.categorical(
                logits=tf.math.log(tf.expand_dims(input_tensor, 0)),
                num_samples=num_samples
            )[0]

    return {"sampled_indices": sampled_indices.numpy()}

def main():
    input_data = {
        "input": np.array([0, 10, 3, 0], dtype=np.float32),
        "num_samples": 2,
        "replacement": False,
    }

    torch_result = torch_multinomial_version(input_data)
    tf_result = tensorflow_multinomial_version(input_data)

    print("Torch sampled indices:", torch_result["sampled_indices"])
    print("TensorFlow sampled indices:", tf_result["sampled_indices"])

    if np.array_equal(torch_result["sampled_indices"], tf_result["sampled_indices"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()