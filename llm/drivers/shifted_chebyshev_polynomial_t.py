import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    n = torch.tensor(input_dict["n"]).int()
    x = torch.tensor(input_dict["x"])
    alpha = input_dict.get("alpha", 0.0)
    
    if not cpu:
        n = n.cuda()
        x = x.cuda()
    
    result = torch.special.shifted_chebyshev_polynomial_t(x, n, alpha=alpha)
    
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
        n = tf.constant(input_dict["n"])
        x = tf.constant(input_dict["x"])
        alpha = input_dict.get("alpha", 0.0)

        n_float = tf.cast(n, dtype=tf.float32)

        def shifted_chebyshev_polynomial_t(n, x, alpha):
          def chebyshev_recurrence(n, x, alpha):
            t0 = tf.ones_like(x)
            t1 = x + alpha
            if n == 0:
              return t0
            elif n == 1:
              return t1
            else:
              for i in range(2, n + 1):
                ti = 2 * 2.0 * (x + alpha) * t1 - t0
                t0 = t1
                t1 = ti
              return ti

          result = tf.cond(
              tf.equal(n, 0),
              lambda: tf.ones_like(x),
              lambda: tf.cond(
                  tf.equal(n, 1),
                  lambda: x + alpha,
                  lambda: chebyshev_recurrence(n, x, alpha)
              )
          )
          return result
        
        result = shifted_chebyshev_polynomial_t(n, x, alpha)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "n": np.array(3, dtype=np.int32),
        "x": np.array(0.5, dtype=np.float32),
        "alpha": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "n": np.array(0, dtype=np.int32),
        "x": np.array(0.5, dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "n": np.array(1, dtype=np.int32),
        "x": np.array(0.5, dtype=np.float32),
        "alpha": 0.2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()