import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    mode = input_dict.get("mode", "reduced")

    if not cpu:
        A = A.cuda()
    
    Q, R = torch.linalg.qr(A, mode=mode)
    
    if not cpu:
        Q = Q.cpu()
        R = R.cpu()
    
    return {"Q": Q.numpy(), "R": R.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        mode = input_dict.get("mode", "reduced")
        
        if mode == "reduced":
            Q, R = tf.linalg.qr(A, full_matrices=False)
        elif mode == "complete":
            Q, R = tf.linalg.qr(A, full_matrices=True)
        elif mode == "r":
            Q, R = tf.linalg.qr(A, full_matrices=False)
            Q = tf.constant([])

        Q = Q.numpy()
        R = R.numpy()

    return {"Q": Q, "R": R}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["Q"], tf_result["Q"], atol=A_TOL), "Q results do not match"
    assert np.allclose(torch_result["R"], tf_result["R"], atol=A_TOL), "R results do not match"

    input_data = {
        "A": np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float32),
        "mode": 'r'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["Q"], tf_result["Q"], atol=A_TOL), "Q results do not match"
    assert np.allclose(torch_result["R"], tf_result["R"], atol=A_TOL), "R results do not match"

    input_data = {
        "A": np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float32),
        "mode": 'complete'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["Q"], tf_result["Q"], atol=A_TOL), "Q results do not match"
    assert np.allclose(torch_result["R"], tf_result["R"], atol=A_TOL), "R results do not match"

    print("Success")

if __name__ == "__main__":
    main()