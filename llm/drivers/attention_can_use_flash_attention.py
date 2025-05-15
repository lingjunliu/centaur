import numpy as np
import torch
from torch.nn.attention import can_use_flash_attention

def torch_version(input_dict, cpu=True):
    q = torch.tensor(input_dict["q"])
    k = torch.tensor(input_dict["k"])
    v = torch.tensor(input_dict["v"])
    mask = torch.tensor(input_dict["mask"]) if "mask" in input_dict else None
    dropout_p = 0.0
    is_causal = False

    if not cpu:
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
        if mask is not None:
            mask = mask.cuda()

    params = torch._C._SDPAParams(q, k, v, mask, dropout_p, is_causal, False)

    result = can_use_flash_attention(params, False)

    if not cpu:
        result = result

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    q = tf.constant(input_dict["q"])
    k = tf.constant(input_dict["k"])
    v = tf.constant(input_dict["v"])
    mask = tf.constant(input_dict["mask"]) if "mask" in input_dict else None

    can_use = True
    if q.dtype != tf.float16 and q.dtype != tf.float32 and q.dtype != tf.float64:
        can_use = False
    q_shape = q.shape
    if len(q_shape) != 4:
        can_use = False

    # Flash Attention v1 only supports causal masking (2D masks)
    if mask is not None:
      mask_shape = mask.shape
      if len(mask_shape) != 2 and len(mask_shape) != 3:
        can_use = False

    return {"result": can_use}

def main():
    A_TOL = 0.01

    input_data = {
        "q": np.random.randn(2, 4, 32, 64).astype(np.float32),
        "k": np.random.randn(2, 4, 32, 64).astype(np.float32),
        "v": np.random.randn(2, 4, 32, 64).astype(np.float32),
        "mask": np.random.randint(0, 2, size=(2, 32)).astype(bool)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()