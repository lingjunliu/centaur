import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    q = input_dict.get("q", None)
    center = input_dict.get("center", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if q is None:
        U, S, V = torch.pca_lowrank(input_tensor, center=center)
    else:
        q_val = min(q, input_tensor.shape[1])
        U, S, V = torch.pca_lowrank(input_tensor, q=q_val, center=center)

    if not cpu:
        U = U.cpu()
        S = S.cpu()
        V = V.cpu()

    return {"U": U.numpy(), "S": S.numpy(), "V": V.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.cast(input_dict["input"], dtype=tf.float32)
        q = input_dict.get("q", None)
        center = input_dict.get("center", True)

        if center:
            mean = tf.reduce_mean(input_tensor, axis=0, keepdims=True)
            centered_tensor = input_tensor - mean
        else:
            centered_tensor = input_tensor

        s, u, v = tf.linalg.svd(centered_tensor)

        if q is not None:
            k = min(q, int(input_tensor.shape[1]))
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

    return {"U": u.numpy(), "S": s.numpy(), "V": v.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(np.abs(torch_result["U"]), np.abs(tf_result["U"]), atol=A_TOL), "U results do not match"
    assert np.allclose(torch_result["S"], tf_result["S"], atol=A_TOL), "S results do not match"
    assert np.allclose(np.abs(torch_result["V"]), np.abs(tf_result["V"]), atol=A_TOL), "V results do not match"

    input_data_q = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "q": 1
    }

    torch_result_q = torch_version(input_data_q)
    tf_result_q = tensorflow_version(input_data_q)

    assert np.allclose(np.abs(torch_result_q["U"]), np.abs(tf_result_q["U"]), atol=A_TOL), "U results do not match (q=1)"
    assert np.allclose(torch_result_q["S"], tf_result_q["S"], atol=A_TOL), "S results do not match (q=1)"
    assert np.allclose(np.abs(torch_result_q["V"]), np.abs(tf_result_q["V"]), atol=A_TOL), "V results do not match (q=1)"

    input_data_nocenter = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "center": False
    }

    torch_result_nocenter = torch_version(input_data_nocenter)
    tf_result_nocenter = tensorflow_version(input_data_nocenter)

    assert np.allclose(np.abs(torch_result_nocenter["U"]), np.abs(tf_result_nocenter["U"]), atol=A_TOL), "U results do not match (no center)"
    assert np.allclose(torch_result_nocenter["S"], tf_result_nocenter["S"], atol=A_TOL), "S results do not match (no center)"
    assert np.allclose(np.abs(torch_result_nocenter["V"]), np.abs(tf_result_nocenter["V"]), atol=A_TOL), "V results do not match (no center)"

    print("Success")

if __name__ == "__main__":
    main()