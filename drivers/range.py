import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    start = input['start']
    end = input['end']
    step = input.get('step', 1)
    requires_grad = input.get('requires_grad', False)

    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Apply to torch.range
    if 'dtype' in input.keys():
        dtype = torch.tensor(np.array([], dtype=input["dtype"])).dtype
        result = torch.range(start=start, end=end, step=step, device=device, dtype=dtype, requires_grad=requires_grad)
    
    else:
        result = torch.range(start=start, end=end, step=step, device=device, requires_grad=requires_grad)

    if not cpu:
        result = result.cpu()

    return {"range": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    # Unpack input dictionary
    start = input['start']
    end = input['end']
    step = input.get('step', 1)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Apply to TensorFlow equivalent
        if 'dtype' in input.keys():
            dtype = tf.as_dtype(input['dtype'])
            result = tf.range(start=start, limit=end + step, delta=step, dtype=dtype)
        else:
            result = tf.range(start=start, limit=end + step, delta=step)
    
    return {"range": result.numpy()}

def main():
    # Example input
    input_data = {
        "start": 1.0,
        "end": 4.0,
        "step": 0.5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality and print results
    assert np.allclose(torch_result["range"], tf_result["range"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()