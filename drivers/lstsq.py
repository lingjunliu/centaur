import numpy as np

# Even though TensorFlow equivalent for lstsq does not provide "out" parameter
# we will replicate the functionality ignoring "out" the same way as PyTorch.
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["B"])
    A_tensor = torch.tensor(input["A"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        A_tensor = A_tensor.cuda()
    
    # Apply to torch.lstsq
    solution = torch.linalg.lstsq(A_tensor, input_tensor).solution
    
    if not cpu:
        solution = solution.cpu()

    return {"lstsq_solution": solution.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["B"])
        A_tensor = tf.constant(input["A"])
        out = input.get("out", None)  # We will not be using out as this is fixed.

        # Apply to TensorFlow equivalent tf.linalg.lstsq
        # tf.linalg.lstsq does not have "out" parameter functionality
        solution = tf.linalg.lstsq(A_tensor, input_tensor)

        return {"lstsq_solution": solution.numpy()}

def main():
    # Example input
    input_data = {
        "B": np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        "A": np.array([[2., 0.], [0., 2.]], dtype=np.float32),
        "out": None  # Fixed out parameter
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparing results using numpy.allclose for floating point comparisons
    if np.allclose(torch_result["lstsq_solution"], tf_result["lstsq_solution"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()