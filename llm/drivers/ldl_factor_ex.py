import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    L, D, pivots = torch.linalg.ldl_factor_ex(input_tensor)

    if not cpu:
        L = L.cpu()
        D = D.cpu()
        pivots = pivots.cpu()

    return {"L": L.numpy(), "D": D.numpy(), "pivots": pivots.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.linalg import set_diag, matmul

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        A = tf.convert_to_tensor(input_tensor)
        n = tf.shape(A)[0]
        
        if len(A.shape) == 2:
            pivots_tf = tf.range(n)
            L_tf = tf.eye(n)
            D_tf = tf.zeros(n, dtype=A.dtype)
        else:
            raise NotImplementedError("Only support 2D tensors for now.")

        A_np = A.numpy()
        pivots_np = np.arange(n)
        L_np = np.eye(n)
        D_np = np.zeros(n, dtype=A_np.dtype)
        
        for i in range(n):
            if not np.isclose(A_np[i,i], 0.0):
                D_np[i] = A_np[i,i]
                L_np[i+1:, i] = A_np[i+1:, i] / A_np[i,i]
                A_np[i+1:, i+1:] -= np.outer(L_np[i+1:, i], L_np[i+1:, i] * D_np[i])
            else: #Handling zero diagonal elements by skipping row and column
                L_np[i+1:, i] = 0.0
                L_np[i, i+1:] = 0.0
                D_np[i] = 0.0
                

        L_tf = tf.convert_to_tensor(L_np)
        D_tf = tf.convert_to_tensor(D_np)
        pivots_tf = tf.convert_to_tensor(pivots_np)
        
        if not cpu:
            L_tf = tf.identity(L_tf)
            D_tf = tf.identity(D_tf)
            pivots_tf = tf.identity(pivots_tf)

        L_tf = L_tf.numpy()
        D_tf = D_tf.numpy()
        pivots_tf = pivots_tf.numpy()
        
        L_tf = np.tril(L_tf, k=-1) + np.eye(L_tf.shape[0])
    return {"L": L_tf, "D": D_tf, "pivots": pivots_tf}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[4.0, 12.0, -16.0],
                           [12.0, 37.0, -43.0],
                           [-16.0, -43.0, 98.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L Results do not match"
    assert np.allclose(torch_result["D"], tf_result["D"], atol=A_TOL), "D Results do not match"
    assert np.allclose(torch_result["pivots"], tf_result["pivots"], atol=A_TOL), "Pivots Results do not match"

    print("Success")

if __name__ == "__main__":
    main()