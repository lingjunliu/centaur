import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    b = torch.tensor(input_dict["b"])
    LD = torch.tensor(input_dict["LD"])
    pivots = torch.tensor(input_dict["pivots"])

    if not cpu:
        A = A.cuda()
        b = b.cuda()
        LD = LD.cuda()
        pivots = pivots.cuda()

    result = torch.linalg.ldl_solve(LD, pivots, b.unsqueeze(-1)).squeeze(-1)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    A = tf.constant(input_dict["A"], dtype=tf.float32)
    b = tf.constant(input_dict["b"], dtype=tf.float32)
    LD = tf.constant(input_dict["LD"], dtype=tf.float32)
    pivots = tf.constant(input_dict["pivots"])
    pivots = tf.cast(pivots, dtype=tf.int32)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):

        n = A.shape[0]

        x = tf.identity(b)
        x = tf.cast(x, dtype=tf.float32)

        x = tf.reshape(x, shape=(-1, 1))
        
        x_pivoted = tf.identity(x)
        for i in range(n):
            k = pivots[i] - 1
            x_pivoted = tf.tensor_scatter_nd_update(x_pivoted, [[i]], [x[k]])

        x = tf.linalg.triangular_solve(LD, x_pivoted, lower=True)
        x = tf.linalg.triangular_solve(tf.transpose(LD), x, lower=False)

        result = tf.reshape(x, shape=(-1,))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[7., -2., 1.], [-2., 5., -2.], [1., -2., 3.]], dtype=np.float32),
        "b": np.array([11., -5., 5.], dtype=np.float32),
        "LD": np.array([[7., -2., 1.], [0., 4.142857, -1.714286], [0., 0., 2.285714]], dtype=np.float32),
        "pivots": np.array([1, 2, 3], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()