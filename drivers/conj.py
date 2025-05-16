import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.cfloat)

    # Apply torch.conj
    conj_tensor = torch.conj(input_tensor)
    resolved_conj_tensor = conj_tensor.resolve_conj()

    if not cpu:
        resolved_conj_tensor = resolved_conj_tensor.cpu()

    return {"conj_tensor": resolved_conj_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.complex64)

        # Apply TensorFlow equivalent tf.math.conj
        conj_tensor = tf.math.conj(input_tensor)

        return {"conj_tensor": conj_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([-1 + 1j, -2 + 2j, 3 - 3j], dtype=np.complex64),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["conj_tensor"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["conj_tensor"])

    # Compare results using numpy
    if np.array_equal(torch_result["conj_tensor"], tf_result["conj_tensor"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()