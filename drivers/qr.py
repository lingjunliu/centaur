import numpy as np

def torch_qr(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.qr
    Q, R = torch.qr(input_tensor, some=input.get("some", True))

    if not cpu:
        Q = Q.cpu()
        R = R.cpu()

    return {"Q": Q.numpy(), "R": R.numpy()}

def tensorflow_qr(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent using tf.linalg.qr
        Q, R = tf.linalg.qr(input_tensor, full_matrices=not input.get("some", True))

        return {"Q": Q.numpy(), "R": R.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_qr(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_qr(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_Q = torch_result["Q"]
    torch_R = torch_result["R"]
    tf_Q = tf_result["Q"]
    tf_R = tf_result["R"]

    # Checking if both Q and R matrices are equal
    if np.allclose(torch_Q, tf_Q) and np.allclose(torch_R, tf_R):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()