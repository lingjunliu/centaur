import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    b_tensor = torch.tensor(input["b"])
    A_tensor = torch.tensor(input["A"])
    
    upper = input.get("upper", True)
    transpose = input.get("transpose", False)
    unitriangular = input.get("unitriangular", False)

    if not cpu:
        b_tensor = b_tensor.cuda()
        A_tensor = A_tensor.cuda()

    # Apply to torch.triangular_solve
    solution, _ = torch.triangular_solve(b_tensor, A_tensor, upper=upper, transpose=transpose, unitriangular=unitriangular)

    if not cpu:
        solution = solution.cpu()

    return {"triangular_solve_solution": solution.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        b_tensor = tf.constant(input["b"])
        A_tensor = tf.constant(input["A"])

        upper = input.get("upper", True)
        transpose = input.get("transpose", False)
        unitriangular = input.get("unitriangular", False)

        # Apply to TensorFlow equivalent using tf.linalg.triangular_solve
        solution = tf.linalg.triangular_solve(A_tensor, b_tensor, lower=not upper, adjoint=transpose)

        return {"triangular_solve_solution": solution.numpy()}


def main():
    # Example input
    input_data = {
        "b": np.array([[2.0], [1.0], [3.0]], dtype=np.float32),
        "A": np.array([[2.0, 0, 0], [3.0, 1.0, 0], [1.0, 2.0, 2.0]], dtype=np.float32),
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results and print if they are equal or not
    if np.allclose(torch_result["triangular_solve_solution"], tf_result["triangular_solve_solution"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()