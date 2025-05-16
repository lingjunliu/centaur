import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    context_offset = input_dict["context_offset"]
    new_size = input_dict["new_size"]
    alloc = input_dict.get("alloc", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    storage = input_tensor.untyped_storage()
    
    if alloc:
        new_storage = torch.UntypedStorage(new_size)
        
        copy_length = min(new_size, len(storage) - context_offset)
        
        if copy_length > 0:
            source_data = storage[context_offset:context_offset + copy_length]
            new_storage[:copy_length] = source_data
        
        result = torch.tensor(new_storage)

    else:
        if new_size > 0 and len(storage) > context_offset:
          result = torch.tensor(storage[context_offset:context_offset + new_size])
        else:
          result = torch.tensor(np.array([]))
    
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
        context_offset = input_dict["context_offset"]
        new_size = input_dict["new_size"]
        alloc = input_dict.get("alloc", False)

        size = tf.size(input_tensor)

        start = context_offset
        end = tf.minimum(context_offset + new_size, size)
        
        if alloc:
          result = tf.Variable(tf.zeros([new_size], dtype=input_tensor.dtype))
          update_size = tf.minimum(new_size, end - start)
          
          if update_size > 0:
            indices = tf.range(0, update_size)
            updates = tf.slice(input_tensor, [start], [update_size])
            result = tf.tensor_scatter_nd_update(result, tf.expand_dims(indices, axis=1), updates)
          else:
              result = tf.Variable(tf.zeros([0], dtype=input_tensor.dtype))

        else:
            if (end - start) > 0:
              result = tf.slice(input_tensor, [start], [end - start])
            else:
              result = tf.constant(np.array([]))
            
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "context_offset": 1,
        "new_size": 2,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "context_offset": 1,
        "new_size": 5,
        "alloc": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "context_offset": 5,
        "new_size": 2,
        "alloc": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "context_offset": 5,
        "new_size": 0,
        "alloc": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()