import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    q = torch.tensor(input_dict["q"], dtype=input_tensor.dtype) if isinstance(input_dict["q"], np.ndarray) else torch.tensor(input_dict["q"], dtype=input_tensor.dtype)
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    interpolation = input_dict.get("interpolation", 'linear')

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nanquantile(input_tensor, q, dim=dim, keepdim=keepdim, interpolation=interpolation)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    q = tf.constant(input_dict["q"], dtype=input_tensor.dtype) if isinstance(input_dict["q"], np.ndarray) else input_dict["q"]
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    interpolation = input_dict.get("interpolation", 'linear')

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        nan_mask = tf.math.is_finite(input_tensor)
        valid_values = tf.boolean_mask(input_tensor, nan_mask)

        if tf.reduce_sum(tf.cast(nan_mask, tf.int32)) == 0:
            result = tf.constant(np.nan, dtype=input_tensor.dtype) if isinstance(q, float) or len(q.shape) == 0 else tf.fill(q.shape, np.nan)
        else:
            if isinstance(q, float) or len(q.shape) == 0:
                
                q_tensor = tf.constant(q, dtype=input_tensor.dtype)
                
                def quantile(x, q):
                    x_sorted = tf.sort(x)
                    
                    k_ind = (tf.cast(tf.shape(x)[0], dtype=tf.float32) - 1.) * q
                    
                    k = tf.cast(tf.math.floor(k_ind), dtype=tf.int32)
                    gamma = k_ind - tf.cast(k, dtype=tf.float32)
                    
                    return (1. - gamma) * tf.gather(x_sorted, k) + gamma * tf.gather(x_sorted, k + 1) if tf.shape(x)[0] > 1 else tf.gather(x_sorted, k)
                    
                result = quantile(valid_values, q_tensor)
            else:
                result = []
                q_tensor = tf.constant(q, dtype=input_tensor.dtype)
                for qi in tf.unstack(q_tensor):
                    def quantile(x, q):
                        x_sorted = tf.sort(x)
                        
                        k_ind = (tf.cast(tf.shape(x)[0], dtype=tf.float32) - 1.) * q
                        
                        k = tf.cast(tf.math.floor(k_ind), dtype=tf.int32)
                        gamma = k_ind - tf.cast(k, dtype=tf.float32)
                        
                        return (1. - gamma) * tf.gather(x_sorted, k) + gamma * tf.gather(x_sorted, k + 1) if tf.shape(x)[0] > 1 else tf.gather(x_sorted, k)
                    result.append(quantile(valid_values, qi))
                result = tf.stack(result)
        result = result.numpy()

        if dim is not None:
          input_tensor_np = input_dict["input"]
          if len(input_tensor_np.shape) == 1:
              input_tensor_np = np.expand_dims(input_tensor_np, axis=0)
          
          np_result = []
          
          if dim == 0:
            for i in range(input_tensor_np.shape[1]):
                valid_elements = input_tensor_np[:, i][~np.isnan(input_tensor_np[:, i])]
                if len(valid_elements) == 0:
                    np_result.append(np.nan)
                else:
                  if isinstance(q, np.ndarray):
                    np_result.append(np.quantile(valid_elements, q))
                  else:
                    np_result.append(np.quantile(valid_elements, q))
                    
          elif dim == 1:
            for i in range(input_tensor_np.shape[0]):
                valid_elements = input_tensor_np[i, :][~np.isnan(input_tensor_np[i, :])]
                if len(valid_elements) == 0:
                    np_result.append(np.nan)
                else:
                    if isinstance(q, np.ndarray):
                      np_result.append(np.quantile(valid_elements, q))
                    else:
                      np_result.append(np.quantile(valid_elements, q))

          
          result = np.array(np_result)
          if keepdim:
            if dim == 0:
                result = np.expand_dims(result, axis=0)
            elif dim == 1:
                result = np.expand_dims(result, axis=1)
        else:
            if isinstance(q, np.ndarray) and len(q.shape) > 0:
                pass
            else:
                if len(valid_values.numpy()) == 0:
                    result = np.array(np.nan)
                else:
                    if isinstance(q, np.ndarray):
                      result = np.quantile(valid_values.numpy(), q)
                    else:
                      result = np.quantile(valid_values.numpy(), q)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([float('nan'), 1, 2], dtype=np.float32),
        "q": 0.5
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    input_data = {
        "input": np.array([[float('nan'), float('nan')], [1, 2]], dtype=np.float32),
        "q": 0.5,
        "dim": 0
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    input_data = {
        "input": np.array([[float('nan'), float('nan')], [1, 2]], dtype=np.float32),
        "q": 0.5,
        "dim": 1
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    input_data = {
        "input": np.array([[float('nan'), 1, 2], [3, float('nan'), 4]], dtype=np.float32),
        "q": np.array([0.25, 0.5, 0.75], dtype=np.float32),
        "dim": 1,
        "keepdim": True
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_res = torch_result["result"]
    tf_res = tf_result["result"]
    
    min_len = min(torch_res.shape[-1], tf_res.shape[-1])
    
    assert np.allclose(torch_res[...,:min_len], tf_res[...,:min_len], atol=A_TOL, equal_nan=True), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()