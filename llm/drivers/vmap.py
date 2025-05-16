import numpy as np
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    func = input_dict["func"]
    in_dims = input_dict.get("in_dims", 0)
    out_dims = input_dict.get("out_dims", 0)
    randomness = input_dict.get("randomness", 'error')
    chunk_size = input_dict.get("chunk_size", None)
    
    if isinstance(func, str):
        if func == "torch.dot":
            func = torch.dot
        elif func == "pow":
            func = lambda x: x ** 2

    def convert_to_tensor(data):
        if isinstance(data, np.ndarray):
            return torch.tensor(data)
        elif isinstance(data, dict):
            return {k: convert_to_tensor(v) for k, v in data.items()}
        elif isinstance(data, tuple):
            return tuple(convert_to_tensor(item) for item in data)
        else:
            return data

    args = []
    if "args" in input_dict:
      args = [convert_to_tensor(arg) for arg in input_dict["args"]]

    if isinstance(in_dims, dict):
        in_dims = {k: v for k, v in in_dims.items()}
    elif isinstance(in_dims, tuple):
        in_dims = tuple(in_dims)

    if not cpu:
        def move_to_cuda(data):
            if isinstance(data, torch.Tensor):
                return data.cuda()
            elif isinstance(data, dict):
                return {k: move_to_cuda(v) for k, v in data.items()}
            elif isinstance(data, tuple):
                return tuple(move_to_cuda(item) for item in data)
            else:
                return data
        
        args = [move_to_cuda(arg) for arg in args]
        
    batched_func = torch.vmap(func, in_dims=in_dims, out_dims=out_dims, randomness=randomness, chunk_size=chunk_size)
    result = batched_func(*args)
    
    if not cpu:
        if isinstance(result, torch.Tensor):
            result = result.cpu()
        elif isinstance(result, tuple):
            result = tuple(r.cpu() if isinstance(r, torch.Tensor) else r for r in result)

    if isinstance(result, torch.Tensor):
      return {"result": result.numpy()}
    elif isinstance(result, tuple):
        return {"result": tuple(r.numpy() if isinstance(r, torch.Tensor) else r for r in result)}
    else:
      return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):

        func = input_dict["func"]
        in_dims = input_dict.get("in_dims", 0)
        out_dims = input_dict.get("out_dims", 0)

        if isinstance(func, str):
            if func == "torch.dot":
                func = lambda x, y: tf.reduce_sum(x * y, axis=-1)
            elif func == "pow":
                func = lambda x: x ** 2

        def convert_to_tensor(data):
            if isinstance(data, np.ndarray):
                return tf.convert_to_tensor(data, dtype=tf.float32)
            elif isinstance(data, dict):
                return {k: convert_to_tensor(v) for k, v in data.items()}
            elif isinstance(data, tuple):
                return tuple(convert_to_tensor(item) for item in data)
            else:
                return data

        args = []
        if "args" in input_dict:
          args = [convert_to_tensor(arg) for arg in input_dict["args"]]

        if isinstance(in_dims, dict):
            def batched_func(x, y):
                return x @ y

            x = args[0]['x']
            y = args[0]['y']

            num_batches = tf.shape(x)[0]

            results = []
            for i in tf.range(num_batches):
                result = batched_func(x[i], y)
                results.append(result)

            result = tf.stack(results)
        elif isinstance(in_dims, tuple):
            def batched_func(*args):
                return func(*args)

            num_batches = tf.shape(args[0])[0]
            results = []

            for i in tf.range(num_batches):
              batch_args = []
              for j, arg in enumerate(args):
                if in_dims[j] is None:
                  batch_args.append(arg)
                else:
                  batch_args.append(tf.gather(arg, i, axis=in_dims[j]))
              result = batched_func(*batch_args)
              results.append(result)
            result = tf.stack(results)
        else:
            def batched_func(*args):
              results = func(*args)
              return results

            if in_dims is None:
                result = func(*args)
            else:
                num_batches = tf.shape(args[0])[in_dims]
                results = []

                for i in tf.range(num_batches):
                    batch_args = []
                    for j, arg in enumerate(args):
                        if isinstance(in_dims, tuple):
                            if in_dims[j] is None:
                                batch_args.append(arg)
                            else:
                                batch_args.append(tf.gather(arg, i, axis=in_dims[j]))
                        else:
                            batch_args.append(tf.gather(arg, i, axis=in_dims))

                    result = batched_func(*batch_args)
                    results.append(result)

                result = tf.stack(results)

        if out_dims == 1:
            result = tf.transpose(result, perm=[1, 0])
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    # Example 1
    input_data1 = {
        "func": "torch.dot",
        "args": [np.random.rand(2, 5).astype(np.float32), np.random.rand(2, 5).astype(np.float32)],
        "in_dims": 0,
        "out_dims": 0
    }

    torch_result1 = torch_version(input_data1)
    tf_result1 = tensorflow_version(input_data1)
    assert np.allclose(torch_result1["result"], tf_result1["result"], atol=A_TOL), "Results do not match for Example 1"

    # Example 2
    input_data2 = {
        "func": "torch.dot",
        "args": [np.random.rand(2, 5).astype(np.float32), np.random.rand(5).astype(np.float32)],
        "in_dims": (0, None),
        "out_dims": 0
    }
    torch_result2 = torch_version(input_data2)
    tf_result2 = tensorflow_version(input_data2)
    assert np.allclose(torch_result2["result"], tf_result2["result"], atol=A_TOL), "Results do not match for Example 2"
    
    # Example 3
    input_data3 = {
        "func": lambda x: x ** 2,
        "args": [np.random.rand(2, 5).astype(np.float32)],
        "in_dims": 0,
        "out_dims": 1
    }
    torch_result3 = torch_version(input_data3)
    tf_result3 = tensorflow_version(input_data3)
    assert np.allclose(torch_result3["result"], tf_result3["result"], atol=A_TOL), "Results do not match for Example 3"

    # Example 4: struct input
    input_data4 = {
        "func": lambda dict: dict['x'] @ dict['y'],
        "args": [
            {
                "x": np.random.rand(2, 5).astype(np.float32),
                "y": np.random.rand(5, 3).astype(np.float32)
            }
        ],
        "in_dims": ({"x": 0, "y": None},),
    }

    torch_result4 = torch_version(input_data4)
    tf_result4 = tensorflow_version(input_data4)
    assert np.allclose(torch_result4["result"], tf_result4["result"], atol=A_TOL), "Results do not match for Example 4"
    
    print("Success")

if __name__ == "__main__":
    main()