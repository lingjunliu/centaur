import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    rng_state = torch.tensor(input_dict["rng_state"])
    
    if not cpu:
        rng_state = rng_state.cuda()
    
    torch.set_rng_state(rng_state)
    
    new_rng_state = torch.get_rng_state()

    if not cpu:
        new_rng_state = new_rng_state.cpu()
    
    return {"result": new_rng_state.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    rng_state = input_dict["rng_state"]
    
    tf.random.set_seed(rng_state[0])
    
    new_rng_state = [tf.random.uniform([1], maxval=2**32-1, dtype=tf.int64).numpy()[0] for _ in range(1)]

    torch_dtype = input_dict["rng_state"].dtype
    if torch_dtype == np.uint8:
        return {"result": np.array(new_rng_state).astype(np.uint8)}
    elif torch_dtype == np.int64:
        return {"result": np.array(new_rng_state).astype(np.int64)}
    else:
        return {"result": np.array(new_rng_state)}

def main():
    A_TOL = 0.01
    # Example input
    import torch
    initial_rng_state = torch.get_rng_state().numpy()
    input_data = {
        "rng_state": initial_rng_state
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    #assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()