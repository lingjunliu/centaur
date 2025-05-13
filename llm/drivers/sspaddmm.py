import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"]).to_sparse()
    mat1 = torch.tensor(input_dict["mat1"]).to_sparse()
    mat2 = torch.tensor(input_dict["mat2"])
    beta = input_dict.get("beta", 1.0)
    alpha = input_dict.get("alpha", 1.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        mat1 = mat1.cuda()
        mat2 = mat2.cuda()

    result = torch.sspaddmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)

    if not cpu:
        result = result.cpu()

    return {"result": result.to_dense().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.sparse.from_dense(tf.constant(input_dict["input"]))
        mat1 = tf.sparse.from_dense(tf.constant(input_dict["mat1"]))
        mat2 = tf.constant(input_dict["mat2"])
        beta = input_dict.get("beta", 1.0)
        alpha = input_dict.get("alpha", 1.0)

        mat1_dense = tf.sparse.to_dense(mat1)
        input_dense = tf.sparse.to_dense(input_tensor)
        
        mat_mul = tf.matmul(mat1_dense, mat2)
        result = beta * input_dense + alpha * mat_mul

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0, 1], [2, 0]], dtype=np.float32),
        "mat1": np.array([[1, 0], [0, 2]], dtype=np.float32),
        "mat2": np.array([[3, 4], [5, 6]], dtype=np.float32),
        "beta": 0.5,
        "alpha": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()