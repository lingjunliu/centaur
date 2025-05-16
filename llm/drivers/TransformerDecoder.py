import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):

    tgt = torch.tensor(input_dict["tgt"])
    memory = torch.tensor(input_dict["memory"])
    tgt_mask = torch.tensor(input_dict["tgt_mask"]) if "tgt_mask" in input_dict else None
    memory_mask = torch.tensor(input_dict["memory_mask"]) if "memory_mask" in input_dict else None
    tgt_key_padding_mask = torch.tensor(input_dict["tgt_key_padding_mask"]) if "tgt_key_padding_mask" in input_dict else None
    memory_key_padding_mask = torch.tensor(input_dict["memory_key_padding_mask"]) if "memory_key_padding_mask" in input_dict else None
    
    decoder_layer = input_dict["decoder_layer"]
    num_layers = input_dict.get("num_layers", 6)
    norm = input_dict.get("norm", None)

    if not cpu:
        tgt = tgt.cuda()
        memory = memory.cuda()
        if tgt_mask is not None:
            tgt_mask = tgt_mask.cuda()
        if memory_mask is not None:
            memory_mask = memory_mask.cuda()
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.cuda()
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.cuda()

    decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers, norm)
    result = decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                     tgt_key_padding_mask=tgt_key_padding_mask,
                     memory_key_padding_mask=memory_key_padding_mask)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):

    tgt = tf.constant(input_dict["tgt"])
    memory = tf.constant(input_dict["memory"])
    tgt_mask = tf.constant(input_dict["tgt_mask"]) if "tgt_mask" in input_dict else None
    memory_mask = tf.constant(input_dict["memory_mask"]) if "memory_mask" in input_dict else None
    tgt_key_padding_mask = tf.constant(input_dict["tgt_key_padding_mask"]) if "tgt_key_padding_mask" in input_dict else None
    memory_key_padding_mask = tf.constant(input_dict["memory_key_padding_mask"]) if "memory_key_padding_mask" in input_dict else None
    
    decoder_layer = input_dict["decoder_layer"]
    num_layers = input_dict.get("num_layers", 6)
    norm = input_dict.get("norm", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        
        x = tgt
        for _ in range(num_layers):
            x = decoder_layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        
        if norm is not None:
            x = norm(x)

        result = x.numpy()
    
    return {"result": result}

class DummyDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.device = device
        self.dtype = dtype
        self.self_attn = DummySelfAttention(batch_first=batch_first)
        self.cross_attn = DummyCrossAttention()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                 tgt_key_padding_mask=None, memory_key_padding_mask=None,
                 tgt_is_causal=False, memory_is_causal=False):
        return tgt + memory

class DummySelfAttention(torch.nn.Module):
    def __init__(self, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        return query

class DummyCrossAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        return query

class DummyNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device
        self.dtype = dtype

    def forward(self, input):
        return input

def main():
    A_TOL = 0.01
    # Example input
    d_model = 512
    nhead = 8
    tgt_len = 10
    memory_len = 10
    batch_size = 32
    batch_first = False

    input_data = {
        "tgt": np.random.rand(tgt_len, batch_size, d_model).astype(np.float32),
        "memory": np.random.rand(memory_len, batch_size, d_model).astype(np.float32),
        "tgt_mask": np.zeros((tgt_len, tgt_len)).astype(np.float32),
        "memory_mask": np.zeros((tgt_len, memory_len)).astype(np.float32),
        "tgt_key_padding_mask": np.zeros((batch_size, tgt_len)).astype(np.float32),
        "memory_key_padding_mask": np.zeros((batch_size, memory_len)).astype(np.float32),
        "decoder_layer": DummyDecoderLayer(d_model=d_model, nhead=nhead, batch_first=batch_first),
        "num_layers": 2,
        "norm": DummyNorm(d_model)
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()