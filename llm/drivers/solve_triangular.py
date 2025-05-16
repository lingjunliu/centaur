import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    b = torch.tensor(input_dict["b"])
    upper = input_dict.get("upper", True)
    transpose = input_dict.get("transpose", False)
    unitriangular = input_dict.get("unitriangular", False)

    if not cpu:
        A = A.cuda()
        b = b.cuda()
    
    result = torch.linalg.solve_triangular(A, b, upper=upper, unitriangular=unitriangular)
    if transpose:
        result = torch.linalg.solve_triangular(A.transpose(-1, -2), b, upper=not upper, unitriangular=unitriangular)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    A_np = input_dict["A"]
    b_np = input_dict["b"]
    upper = input_dict.get("upper", True)
    transpose = input_dict.get("transpose", False)
    unitriangular = input_dict.get("unitriangular", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A = tf.constant(A_np)
        b = tf.constant(b_np)
        
        if transpose:
            A = tf.linalg.transpose(A)

        A = tf.cast(A, dtype=tf.float32)
        b = tf.cast(b, dtype=tf.float32)
        
        result = tf.linalg.triangular_solve(A, b, lower=(not upper), adjoint=False)
        
        if unitriangular:
           n = A_np.shape[0]
           identity = tf.eye(n, dtype=tf.float32)

           if upper:
              mask = tf.linalg.band_part(tf.ones_like(A), 0, -1)
           else:
              mask = tf.linalg.band_part(tf.ones_like(A), -1, 0)

           A_without_diag = A * (1-mask) + mask*identity
           result = tf.linalg.triangular_solve(A_without_diag, b, lower=(not upper), adjoint=False)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[2, 0, 0], [1, 3, 0], [0, 1, 4]], dtype=np.float32),
        "b": np.array([1, 2, 3], dtype=np.float32),
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]], dtype=np.float32),
        "b": np.array([7, 11, 5], dtype=np.float32),
        "upper": True,
        "transpose": False,
        "unitriangular": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[1, 0, 0], [2, 1, 0], [3, 4, 1]], dtype=np.float32),
        "b": np.array([1, 2, 3], dtype=np.float32),
        "upper": False,
        "transpose": True,
        "unitriangular": False
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")