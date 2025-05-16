import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"]) if isinstance(input_dict["other"], np.ndarray) else input_dict["other"]
    rounding_mode = input_dict.get("rounding_mode", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(other_tensor, torch.Tensor):
          other_tensor = other_tensor.cuda()

    if rounding_mode is not None:
      result = torch.divide(input_tensor, other_tensor, rounding_mode=rounding_mode)
    else:
      result = torch.divide(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        other_tensor = tf.constant(input_dict["other"]) if isinstance(input_dict["other"], np.ndarray) else input_dict["other"]
        rounding_mode = input_dict.get("rounding_mode", None)

        if rounding_mode is not None:
            if rounding_mode == 'trunc':
                result = tf.math.floor(tf.divide(input_tensor, other_tensor))
                result = tf.where(tf.greater_equal(result, 0), tf.math.floor(tf.divide(input_tensor, other_tensor)), tf.math.ceil(tf.divide(input_tensor, other_tensor)))
            elif rounding_mode == 'floor':
                result = tf.math.floor(tf.divide(input_tensor, other_tensor))
            else:
                raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")

        else:
            result = tf.divide(input_tensor, other_tensor)
            
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data_float = {
        "input": np.array([2.0, 4.0, 6.0], dtype=np.float32),
        "other": np.array([0.5, 2.0, 3.0], dtype=np.float32)
    }

    input_data_int = {
        "input": np.array([5, 10, 15], dtype=np.int32),
        "other": np.array([2, 3, 4], dtype=np.int32)
    }
    
    input_data_scalar = {
        "input": np.array([2.0, 4.0, 6.0], dtype=np.float32),
        "other": 2.0
    }

    input_data_rounding = {
        "input": np.array([5.0, 10.0, 15.0], dtype=np.float32),
        "other": np.array([2.3, 3.2, 4.5], dtype=np.float32),
        "rounding_mode": 'floor'
    }

    input_data_rounding_trunc = {
        "input": np.array([5.0, 10.0, 15.0], dtype=np.float32),
        "other": np.array([2.3, 3.2, 4.5], dtype=np.float32),
        "rounding_mode": 'trunc'
    }
    
    def test_case(input_data):
      torch_result = torch_version(input_data)
      tf_result = tensorflow_version(input_data)
      assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    test_case(input_data_float)
    test_case(input_data_int)
    test_case(input_data_scalar)
    test_case(input_data_rounding)
    test_case(input_data_rounding_trunc)


    print("Success")

if __name__ == "__main__":
    main()