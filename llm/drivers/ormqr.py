import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    tau_tensor = torch.tensor(input_dict["tau"])
    other_tensor = torch.tensor(input_dict["other"])
    left = input_dict.get("left", True)
    transpose = input_dict.get("transpose", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        tau_tensor = tau_tensor.cuda()
        other_tensor = other_tensor.cuda()

    result = torch.ormqr(input_tensor, tau_tensor, other_tensor, left=left, transpose=transpose)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_np = input_dict["input"]
    tau_np = input_dict["tau"]
    other_np = input_dict["other"]
    left = input_dict.get("left", True)
    transpose = input_dict.get("transpose", False)
    
    mn = input_np.shape[-2]
    k = input_np.shape[-1]
    m = other_np.shape[-2]
    n = other_np.shape[-1]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_np)
        tau_tensor = tf.constant(tau_np)
        other_tensor = tf.constant(other_np)

        def householder_transform(v, tau, x):
            v = tf.cast(v, dtype=x.dtype)
            tau = tf.cast(tau, dtype=x.dtype)
            inner = tf.reduce_sum(v * x, axis=-2, keepdims=True)
            w = x - tau * inner * v
            return w

        Q = tf.eye(mn, dtype=input_tensor.dtype)
        
        for i in range(min(mn, k)):
            v_upper = tf.zeros_like(input_tensor[..., :i, i])
            v_mid = tf.expand_dims(tf.ones_like(input_tensor[..., i, i]), axis=-1)
            v_lower = input_tensor[..., (i+1):, i]

            if i > 0 and (input_tensor.shape[-2] - i -1) >0:
                v = tf.concat([v_upper, v_mid, v_lower], axis=-2)
            elif i > 0 and (input_tensor.shape[-2] - i -1) == 0 :
                v = tf.concat([v_upper, v_mid], axis=-2)
            else:
                v = tf.concat([v_mid, v_lower], axis=-2)

            padding = [[0, 0], [i, mn - v.shape[-2] - i], [0, 0]]

            Vi = tf.pad(v, padding)

            Q = householder_transform(Vi, tau_tensor[..., i], Q)
        
        if transpose:
            Q = tf.linalg.adjoint(Q)
            
        if left:
            result = Q @ other_tensor
        else:
            result = other_tensor @ Q
            
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[1.0, 0.0], [2.0, 1.0]]], dtype=np.float32),
        "tau": np.array([[1.0, 1.0]], dtype=np.float32),
        "other": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        "left": True,
        "transpose": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()