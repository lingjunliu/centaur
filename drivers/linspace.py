import numpy as np

def torch_version(input, cpu=True):
    import torch
    
    # Unpack input dictionary
    start = input["start"]
    end = input["end"]
    steps = input["steps"]
    device = torch.device("cpu") if cpu else torch.device("cuda")

    with torch.no_grad():
        result = torch.linspace(start=start, end=end, steps=steps, device=device)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        start = input["start"]
        end = input["end"]
        steps = input["steps"]

        # Using TensorFlow's linspace equivalent
        result = tf.linspace(start=start, stop=end, num=steps)
        
    return {"result": result.numpy()}

def main():
    # Example input
    input_data = {
        "start": 0.0,
        "end": 10.0,
        "steps": 5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare results
    assert np.allclose(torch_result["result"], tf_result["result"]), "Results do not match"
    if np.array_equal(torch_result["result"], tf_result["result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()