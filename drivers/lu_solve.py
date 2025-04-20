import numpy as np
from numpy.linalg import solve

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    b = torch.tensor(input["b"])
    LU_data = torch.tensor(input["LU_data"])
    LU_pivots = torch.tensor(input["LU_pivots"], dtype=torch.int32)

    if not cpu:
        b = b.cuda()
        LU_data = LU_data.cuda()
        LU_pivots = LU_pivots.cuda()

    # Apply to torch.lu_solve
    solution = torch.lu_solve(b, LU_data, LU_pivots)
    
    if cpu == False:
        solution = solution.cpu()

    return {"lu_solve_solution": solution.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        b = tf.constant(input["b"])
        LU_data = tf.constant(input["LU_data"])
        LU_pivots = tf.constant(input["LU_pivots"], dtype=tf.int32)

        # Tensorflow equivalent of LU solve using numpy
        L = np.tril(LU_data.numpy(), k=-1) + np.eye(LU_data.shape[-1], dtype=LU_data.dtype.as_numpy_dtype)
        U = np.triu(LU_data.numpy())
        
        # Apply the permutations to the right-hand side
        b = np.copy(b.numpy())
        for i, pivot in enumerate(LU_pivots.numpy()):
            b[[i, pivot-1]] = b[[pivot-1, i]]
        
        y = solve(L, b)
        solution = solve(U, y)

        return {"lu_solve_solution": solution}

def main():
    # Example input
    input_data = {
        "b": np.array([[3.0], [4.0]], dtype=np.float32),
        "LU_data": np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32),
        "LU_pivots": np.array([1, 2], dtype=np.int32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure that the results are approximately equal
    if np.allclose(torch_result["lu_solve_solution"], tf_result["lu_solve_solution"], rtol=1e-05, atol=1e-08):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()