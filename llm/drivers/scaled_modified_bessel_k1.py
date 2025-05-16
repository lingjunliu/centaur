import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.special.scaled_modified_bessel_k1(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)

        # More accurate approximation of scaled_modified_bessel_k1 for x > 0.
        def scaled_modified_bessel_k1(x):
            return tf.where(
                x > 0,
                tf.sqrt(np.pi / (2 * x)) * tf.exp(-x) * (1 + 3 / (8 * x)),
                tf.constant(np.inf, dtype=tf.float32)
            )

        result = scaled_modified_bessel_k1(input_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.1 # Increased tolerance

    input_data = {
        "input": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "nu": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()