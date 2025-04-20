import numpy as np

def torch_logspace(input, cpu=True):
    import torch
    
    # Set the device
    device = torch.device('cpu' if cpu else 'cuda')
    
    start = input["start"]
    if isinstance(start, np.ndarray):
        start = torch.tensor(start)
        if not cpu:
            start = start.cuda()
        
    
    end = input["end"]
    if isinstance(end, np.ndarray):
        end = torch.tensor(end)
        if not cpu:
            end = end.cuda()
    
    steps = input["steps"]
    base = input.get("base", 10.0)
    
    dtype = input.get("dtype", None)
    
    # Generate tensor
    result = torch.logspace(start, end, steps, base=base, dtype=dtype, device=device)
    
    return {"logspace_result": result.cpu().numpy()}

def tensorflow_logspace(input, cpu=True):
    import tensorflow as tf

    # Set the device
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        start = input["start"]
        end = input["end"]
        steps = input["steps"]
        base = input.get("base", 10.0)

        # Generate tensor
        result = tf.convert_to_tensor(np.logspace(start, end, steps, base=base))

        return {"logspace_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "start": 0.1,
        "end": 1.0,
        "steps": 5,
        "base": 10.0
    }

    # Torch example
    torch_result = torch_logspace(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_logspace(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality
    assert np.allclose(torch_result["logspace_result"], tf_result["logspace_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()