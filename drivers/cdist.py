import numpy as np

def torch_version(input, cpu=True):
    import torch

    x1_tensor = torch.tensor(input["x1"])
    x2_tensor = torch.tensor(input["x2"])
    p = input.get("p", 2.0)
    compute_mode = input.get("compute_mode", 'use_mm_for_euclid_dist_if_necessary')

    if not cpu:
        x1_tensor = x1_tensor.cuda()
        x2_tensor = x2_tensor.cuda()

    result = torch.cdist(x1_tensor, x2_tensor, p=p, compute_mode=compute_mode).cpu().numpy()

    return {"result": result}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        x1_tensor = tf.constant(input["x1"])
        x2_tensor = tf.constant(input["x2"])
        p = input.get("p", 2.0)
        
        # Expand dimensions to allow broadcasting
        x1_exp = tf.expand_dims(x1_tensor, 2)  # (B, P, 1, M)
        x2_exp = tf.expand_dims(x2_tensor, 1)  # (B, 1, R, M)
        
        if p == float('inf'):
            # Inf-norm
            diff = tf.abs(x1_exp - x2_exp)
            result = tf.reduce_max(diff, axis=-1)  # Max over the M dimension
        else:
            # p-norm
            diff = tf.abs(x1_exp - x2_exp)
            diff_pow = tf.pow(diff, p)
            sum_diff_pow = tf.reduce_sum(diff_pow, axis=-1)  # Sum over the M dimension
            result = tf.pow(sum_diff_pow, 1/p)  # Take p-th root

        result = result.numpy()

    return {"result": result}

def main():
    input_data = {
        "x1": np.array([[[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]]], dtype=np.float32),  # shape (B, P, M)
        "x2": np.array([[[-2.1763, -0.4713], [-0.6986, 1.3702]]], dtype=np.float32),  # shape (B, R, M)
        "p": 2.0,
        "compute_mode": 'use_mm_for_euclid_dist_if_necessary'
    }

    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    are_equal = np.allclose(torch_result["result"], tf_result["result"], rtol=1e-5, atol=1e-5)
    
    if are_equal:
        print("equal")
    else:
        print("not equal")
        
    assert are_equal, "The results from PyTorch and TensorFlow are not equal!"

if __name__ == "__main__":
    main()