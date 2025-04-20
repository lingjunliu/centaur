import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    x = torch.tensor(input["input"])
    N = input.get("N", x.size(0))
    increasing = input.get("increasing", False)

    # Calculate Vandermonde matrix using torch.vander
    vander_matrix = torch.vander(x, N=N, increasing=increasing)

    if not cpu:
        vander_matrix = vander_matrix.cpu()

    return {"vander_matrix": vander_matrix.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        x = input["input"]
        N = input.get("N", len(x))
        increasing = input.get("increasing", False)

        # Calculate the Vandermonde matrix
        def vander(x, N, increasing):
            x = tf.convert_to_tensor(x)
            vander_matrix = tf.math.pow(tf.expand_dims(x, axis=-1), tf.range(N))
            if not increasing:
                vander_matrix = tf.reverse(vander_matrix, axis=[-1])
            return vander_matrix

        vander_matrix = vander(x, N, increasing)
        
        return {"vander_matrix": vander_matrix.numpy()}

def main():
    # Example input
    input_data = {
        "input": [1.0, 2.0, 3.0, 5.0],  # Ensure the input is a list of float
        "N": 3,
        "increasing": True
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality of results
    torch_vander = torch_result["vander_matrix"]
    tf_vander = tf_result["vander_matrix"]

    if np.allclose(torch_vander, tf_vander):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()