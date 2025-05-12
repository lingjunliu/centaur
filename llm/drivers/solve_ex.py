import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    B = torch.tensor(input_dict["B"])
    left = input_dict.get("left", True)

    if not cpu:
        A = A.cuda()
        B = B.cuda()
    
    result = torch.linalg.solve_ex(A, B, left=left)[0]

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        B = tf.constant(input_dict["B"])
        left = input_dict.get("left", True)

        if left:
            result = tf.linalg.solve(A, B)
        else:
            result = tf.linalg.solve(tf.transpose(A), tf.transpose(B))
            result = tf.transpose(result)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "B": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "left": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "B": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "left": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()