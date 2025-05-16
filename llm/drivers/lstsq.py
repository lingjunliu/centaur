import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    B = torch.tensor(input_dict["B"])
    rcond = input_dict.get("rcond", None)
    driver = input_dict.get("driver", None)

    if not cpu:
        A = A.cuda()
        B = B.cuda()

    result = torch.linalg.lstsq(A, B, rcond=rcond, driver=driver)

    solution = result.solution
    residuals = result.residuals
    rank = result.rank
    singular_values = result.singular_values

    if not cpu:
        solution = solution.cpu()
        residuals = residuals.cpu()
        rank = rank.cpu()
        singular_values = singular_values.cpu()

    return {
        "solution": solution.numpy(),
        "residuals": residuals.numpy(),
        "rank": rank.numpy(),
        "singular_values": singular_values.numpy()
    }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        B = tf.constant(input_dict["B"])
        rcond = input_dict.get("rcond", None)
        
        if rcond is None:
            rcond = np.finfo(A.dtype.as_numpy_dtype).eps * max(tf.shape(A)[0], tf.shape(A)[1])

        s = tf.linalg.svd(A, compute_uv=False)
        max_s = s[0]
        cutoff = tf.cast(rcond, A.dtype) * tf.cast(max_s, dtype=A.dtype)

        num_valid = tf.reduce_sum(tf.cast(s > cutoff, tf.int32))
        s = tf.where(s > cutoff, s, tf.zeros_like(s))

        singular_values = s.numpy()
        rank = num_valid.numpy()

        s_inv = tf.linalg.diag(tf.where(s > 0, 1 / s, tf.zeros_like(s)))
        
        A_transpose = tf.transpose(A)
        solution = tf.matmul(tf.linalg.pinv(A), B)

        residuals = tf.reduce_sum(tf.square(tf.linalg.norm(tf.matmul(A, solution) - B, axis=[0,1])))

        solution = solution.numpy()
        residuals = residuals.numpy()


    return {
        "solution": solution,
        "residuals": residuals,
        "rank": rank,
        "singular_values": singular_values
    }

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "B": np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if torch_result["residuals"].size > 0 and tf_result["residuals"].size > 0:
        assert np.allclose(torch_result["residuals"], tf_result["residuals"], atol=A_TOL), "Residuals results do not match"
    elif torch_result["residuals"].size == 0 and tf_result["residuals"].size == 0:
        pass
    else:
        print("Warning: One of the residuals is empty, cannot compare")

    assert np.allclose(torch_result["solution"], tf_result["solution"], atol=A_TOL), "Solution results do not match"
    assert np.allclose(torch_result["rank"], tf_result["rank"], atol=A_TOL), "Rank results do not match"
    assert np.allclose(torch_result["singular_values"], tf_result["singular_values"], atol=A_TOL), "Singular values results do not match"

    print("Success")

if __name__ == "__main__":
    main()