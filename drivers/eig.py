import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    eigenvectors = input.get("eigenvectors", False)

    # Apply to torch.linalg.eig
    L_complex, V_complex = torch.linalg.eig(input_tensor)
    
    # Separate real and imaginary parts
    eig_vals = torch.view_as_real(L_complex).cpu().numpy()
    
    if eigenvectors:
        eig_vecs = torch.view_as_real(V_complex).cpu().numpy()
    else:
        eig_vecs = None

    if not cpu:
        eig_vals = eig_vals.astype(np.float32)
        if eigenvectors:
            eig_vecs = eig_vecs.astype(np.float32)

    return {
        "eig_vals": eig_vals,
        "eig_vecs": eig_vecs
    }

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply to TensorFlow equivalent
        eig_vals, eig_vecs = tf.linalg.eig(input_tensor)

        eig_vals = np.stack((eig_vals.numpy().real, eig_vals.numpy().imag), axis=-1)

        if input.get("eigenvectors", False):
            eig_vecs = np.stack((eig_vecs.numpy().real, eig_vecs.numpy().imag), axis=-1)
        else:
            eig_vecs = None

        return {
            "eig_vals": eig_vals,
            "eig_vecs": eig_vecs
        }

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "eigenvectors": True,
        "out": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparison
    eig_vals_match = np.allclose(torch_result["eig_vals"], tf_result["eig_vals"], atol=1e-6)
    eig_vecs_match = (torch_result["eig_vecs"] is None or np.allclose(torch_result["eig_vecs"], tf_result["eig_vecs"], atol=1e-6))

    if eig_vals_match and eig_vecs_match:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()