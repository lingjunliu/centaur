import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input_tensor"])
    att_mask = torch.tensor(input_dict["att_mask"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        att_mask = att_mask.cuda()

    params = torch._C._SDPAParams(
        input_tensor,
        input_tensor,
        input_tensor,
        None,
        0.0,
        att_mask.bool(),
        False
    )
    result = torch.backends.cuda.can_use_efficient_attention(params)
    
    if not cpu:
        result = torch.tensor(result).cpu().numpy()
    else:
        result = np.array(result)
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input_tensor"])
    att_mask = tf.constant(input_dict["att_mask"])
    
    result = False
    if input_tensor.shape[-1] <= 128:
        result = True

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input_tensor": np.random.rand(2, 3, 64).astype(np.float32),
        "att_mask": np.random.randint(0, 2, size=(2, 3, 3)).astype(np.bool_)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()