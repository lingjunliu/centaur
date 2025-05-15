import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch import nn

    d_model = input_dict["d_model"]
    nhead = input_dict["nhead"]
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-05)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    bias = input_dict.get("bias", True)

    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
                                                batch_first=batch_first, norm_first=norm_first, bias=bias)

    memory = torch.tensor(input_dict["memory"])
    tgt = torch.tensor(input_dict["tgt"])
    tgt_mask = input_dict.get("tgt_mask", None)
    memory_mask = input_dict.get("memory_mask", None)
    tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask", None)
    memory_key_padding_mask = input_dict.get("memory_key_padding_mask", None)
    tgt_is_causal = input_dict.get("tgt_is_causal", False)
    memory_is_causal = input_dict.get("memory_is_causal", False)

    if tgt_mask is not None:
        tgt_mask = torch.tensor(tgt_mask)
    if memory_mask is not None:
        memory_mask = torch.tensor(memory_mask)
    if tgt_key_padding_mask is not None:
        tgt_key_padding_mask = torch.tensor(tgt_key_padding_mask)
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = torch.tensor(memory_key_padding_mask)
    
    if not cpu:
        memory = memory.cuda()
        tgt = tgt.cuda()
        if tgt_mask is not None:
            tgt_mask = tgt_mask.cuda()
        if memory_mask is not None:
            memory_mask = memory_mask.cuda()
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.cuda()
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.cuda()

        decoder_layer = decoder_layer.cuda()

    out = decoder_layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=tgt_is_causal,
                          memory_is_causal=memory_is_causal)

    if not cpu:
        out = out.cpu()

    return {"result": out.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    d_model = input_dict["d_model"]
    nhead = input_dict["nhead"]
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-05)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    bias = input_dict.get("bias", True)

    memory = tf.constant(input_dict["memory"])
    tgt = tf.constant(input_dict["tgt"])
    tgt_mask = input_dict.get("tgt_mask", None)
    memory_mask = input_dict.get("memory_mask", None)
    tgt_key_padding_mask = input_dict.get("tgt_key_padding_mask", None)
    memory_key_padding_mask = input_dict.get("memory_key_padding_mask", None)
    tgt_is_causal = input_dict.get("tgt_is_causal", False)
    memory_is_causal = input_dict.get("memory_is_causal", False)

    if tgt_mask is not None:
        tgt_mask = tf.constant(tgt_mask)
    if memory_mask is not None:
        memory_mask = tf.constant(memory_mask)
    if tgt_key_padding_mask is not None:
        tgt_key_padding_mask = tf.constant(tgt_key_padding_mask)
    if memory_key_padding_mask is not None:
        memory_key_padding_mask = tf.constant(memory_key_padding_mask)
    
    class TFTransformerDecoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu',
                     layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True):
            super(TFTransformerDecoderLayer, self).__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.dim_feedforward = dim_feedforward
            self.dropout = dropout
            self.activation = activation
            self.layer_norm_eps = layer_norm_eps
            self.batch_first = batch_first
            self.norm_first = norm_first
            self.bias = bias

            self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
            self.multihead_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
            
            self.linear1 = tf.keras.layers.Dense(dim_feedforward, use_bias=bias)
            self.dropout_ff = tf.keras.layers.Dropout(dropout)
            self.linear2 = tf.keras.layers.Dense(d_model, use_bias=bias)
            
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
            self.norm3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
            
            self.dropout1 = tf.keras.layers.Dropout(dropout)
            self.dropout2 = tf.keras.layers.Dropout(dropout)
            self.dropout3 = tf.keras.layers.Dropout(dropout)
            
            if activation == 'relu':
                self.activation_fn = tf.nn.relu
            elif activation == 'gelu':
                self.activation_fn = tf.nn.gelu
            else:
                self.activation_fn = activation

        def call(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                 memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
            
            if self.norm_first:
                tgt_norm = self.norm1(tgt)
                tgt2 = self.self_attn(tgt_norm, tgt_norm, tgt_norm, attention_mask=tgt_mask)
                tgt = tgt + self.dropout1(tgt2)
                
                tgt_norm = self.norm2(tgt)
                tgt2 = self.multihead_attn(tgt_norm, memory, memory, attention_mask=memory_mask)
                tgt = tgt + self.dropout2(tgt2)
                
                tgt_norm = self.norm3(tgt)
                tgt2 = self.linear2(self.dropout_ff(self.activation_fn(self.linear1(tgt_norm))))
                tgt = tgt + self.dropout3(tgt2)
                
            else:
                tgt2 = self.self_attn(tgt, tgt, tgt, attention_mask=tgt_mask)
                tgt = tgt + self.dropout1(tgt2)
                tgt = self.norm1(tgt)
                
                tgt2 = self.multihead_attn(tgt, memory, memory, attention_mask=memory_mask)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
                
                tgt2 = self.linear2(self.dropout_ff(self.activation_fn(self.linear1(tgt))))
                tgt = tgt + self.dropout3(tgt2)
                tgt = self.norm3(tgt)
                
            return tgt

    decoder_layer = TFTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
                                                batch_first=batch_first, norm_first=norm_first, bias=bias)

    if batch_first:
        memory = tf.transpose(memory, perm=[1, 0, 2])
        tgt = tf.transpose(tgt, perm=[1, 0, 2])

    out = decoder_layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=tgt_is_causal,
                          memory_is_causal=memory_is_causal)
    
    if batch_first:
        out = tf.transpose(out, perm=[1, 0, 2])

    return {"result": out.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "d_model": 512,
        "nhead": 8,
        "memory": np.random.rand(20, 32, 512).astype(np.float32),
        "tgt": np.random.rand(20, 32, 512).astype(np.float32),
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "d_model": 512,
        "nhead": 8,
        "memory": np.random.rand(32, 20, 512).astype(np.float32),
        "tgt": np.random.rand(32, 20, 512).astype(np.float32),
        "batch_first": True
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()