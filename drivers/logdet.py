import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Convert input to a PyTorch tensor
    input_tensor = torch.tensor(input["input"])

    # Compute log determinant using PyTorch
    log_det = torch.logdet(input_tensor)

    if not cpu:
        log_det = log_det.cpu()

    return {"logdet": float(log_det.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert input to a TensorFlow tensor
        input_tensor = tf.constant(input["input"])

        # Compute determinant and check its sign
        det = tf.linalg.det(input_tensor)

        # Handle cases where determinant is zero or negative
        log_det = tf.cond(det > 0, lambda: tf.math.log(det), lambda: tf.constant(float('nan')))

        return {"logdet": float(log_det.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Check results
    if np.isnan(torch_result["logdet"]) and np.isnan(tf_result["logdet"]):
        print("equal")
    elif np.isclose(torch_result["logdet"], tf_result["logdet"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()