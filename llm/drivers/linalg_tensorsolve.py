import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    B = torch.tensor(input_dict["B"])
    dims = input_dict.get("dims", None)

    if not cpu:
        A = A.cuda()
        B = B.cuda()

    result = torch.linalg.tensorsolve(A, B, dims=dims)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    A_np = input_dict["A"]
    B_np = input_dict["B"]
    dims = input_dict.get("dims", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(A_np)
        B = tf.constant(B_np)

        B_ndim = len(B.shape)
        A_shape = A.shape
        A_ndim = len(A_shape)

        if dims is not None:
            num_dims = len(dims)
            new_pos = []
            dims_list = list(dims)
            for i in range(A_ndim):
                if i not in dims_list:
                    new_pos.append(i)
            A = tf.transpose(A, perm=dims_list + new_pos)
            A_shape = A.shape  # Update A_shape after transpose

        m = 1
        for i in range(B_ndim):
            m *= A_shape[i]

        n = 1
        for i in range(B_ndim, A_ndim):
            n *= A_shape[i]
            
        B_size = np.prod(B_np.shape)
        B_reshaped = tf.reshape(B, (m, B_size // m))

        A_reshaped = tf.reshape(A, (m, n))
        
        X_reshaped = tf.linalg.solve(A_reshaped, B_reshaped)
        
        X_shape = A_shape[B_ndim:]
        X = tf.reshape(X_reshaped, X_shape)

        result = X.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    A_np = np.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4)).astype(np.float32)
    B_np = np.random.randn(2 * 3, 4).astype(np.float32)
    input_data = {
        "A": A_np,
        "B": B_np,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    A_np = np.random.randn(6, 4, 4, 3, 2).astype(np.float32)
    B_np = np.random.randn(4, 3, 2).astype(np.float32)
    input_data = {
        "A": A_np,
        "B": B_np,
        "dims": (0, 2)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()