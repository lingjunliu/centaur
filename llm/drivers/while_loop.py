import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    loop_fn = input_dict["loop_fn"]
    cond_fn = input_dict["cond_fn"]
    inputs = tuple(torch.tensor(x) for x in input_dict["inputs"])

    if not cpu:
        inputs = tuple(x.cuda() for x in inputs)
    
    i, x = inputs

    while cond_fn(i.item(), x):
        i, x = loop_fn(i.item(), x)
        i = torch.tensor(i)
        if not cpu:
            i = i.cuda()

    result = (i,x)

    if not cpu:
        result = tuple(x.cpu() for x in result)
    
    return {"result": tuple(x.numpy() for x in result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    loop_fn = input_dict["loop_fn"]
    cond_fn = input_dict["cond_fn"]
    inputs = tuple(tf.constant(x) for x in input_dict["inputs"])
    maximum_iterations = input_dict.get("maximum_iterations", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    def tf_cond(i, x):
      return cond_fn(i, x.numpy())

    def tf_loop(i, x):
      i_new, x_new = loop_fn(i, x.numpy())
      return np.array(i_new, dtype=np.int32), np.array(x_new, dtype=np.float32)
    
    with tf.device(device_string):
      i, x = inputs
      if maximum_iterations is None:
          def condition(i,x):
              return tf.py_function(func=tf_cond, inp=[i, x], Tout=tf.bool)
          def body(i, x):
              i_new, x_new = tf.py_function(func=tf_loop, inp=[i, x], Tout=[tf.int32, tf.float32])
              return i_new, x_new
          result = tf.while_loop(condition, body, (i,x))
      else:
          def condition(i, x):
              return tf.logical_and(i < maximum_iterations, tf.py_function(func=tf_cond, inp=[i, x], Tout=tf.bool))

          def body(i, x):
              i_new, x_new = tf.py_function(func=tf_loop, inp=[i, x], Tout=[tf.int32, tf.float32])
              return i+1, x_new

          i = tf.constant(0, dtype=tf.int32)
          _, result = tf.while_loop(condition, body, (i, x))

    return {"result": tuple(x.numpy() for x in result)}

def main():
    A_TOL = 0.01

    def torch_cond(i, x):
        return i < 5 and x.sum() < 10

    def torch_loop(i, x):
        return i + 1, x + 1

    def tf_cond(i, x):
        return i < 5 and np.sum(x) < 10

    def tf_loop(i, x):
        return i + 1, x + 1
    
    input_data = {
        "cond_fn": torch_cond,
        "loop_fn": torch_loop,
        "inputs": (np.array(0), np.array([1, 2], dtype=np.float32)),
        "maximum_iterations": 10
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"][0], tf_result["result"][0], atol=A_TOL), "Results do not match for index 0"
    assert np.allclose(torch_result["result"][1], tf_result["result"][1], atol=A_TOL), "Results do not match for index 1"
    print("Success")

if __name__ == "__main__":
    main()