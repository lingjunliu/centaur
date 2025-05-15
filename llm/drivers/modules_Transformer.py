import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    src = torch.tensor(input_dict["src"])
    tgt = torch.tensor(input_dict["tgt"])
    memory = torch.tensor(input_dict["memory"])
    src_mask = torch.tensor(input_dict["src_mask"]) if "src_mask" in input_dict else None
    tgt_mask = torch.tensor(input_dict["tgt_mask"]) if "tgt_mask" in input_dict else None
    memory_mask = torch.tensor(input_dict["memory_mask"]) if "memory_mask" in input_dict else None
    src_key_padding_mask = torch.tensor(input_dict["src_key_padding_mask"]) if "src_key_padding_mask" in input_dict else None
    tgt_key_padding_mask = torch.tensor(input_dict["tgt_key_padding_mask"]) if "tgt_key_padding_mask" in input_dict else None
    memory_key_padding_mask = torch.tensor(input_dict["memory_key_padding_mask"]) if "memory_key_padding_mask" in input_dict else None

    d_model = input_dict["d_model"]
    nhead = input_dict["nhead"]
    num_encoder_layers = input_dict["num_encoder_layers"]
    num_decoder_layers = input_dict["num_decoder_layers"]
    dim_feedforward = input_dict.get("dim_feedforward", 2048)
    dropout = input_dict.get("dropout", 0.1)
    activation = input_dict.get("activation", "relu")
    custom_encoder = input_dict.get("custom_encoder", None)
    custom_decoder = input_dict.get("custom_decoder", None)
    layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
    batch_first = input_dict.get("batch_first", False)
    norm_first = input_dict.get("norm_first", False)
    
    if not cpu:
        src = src.cuda()
        tgt = tgt.cuda()
        memory = memory.cuda()
        if src_mask is not None:
            src_mask = src_mask.cuda()
        if tgt_mask is not None:
            tgt_mask = tgt_mask.cuda()
        if memory_mask is not None:
            memory_mask = memory_mask.cuda()
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.cuda()
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.cuda()
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.cuda()

    transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, custom_encoder=custom_encoder, custom_decoder=custom_decoder, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
    
    kwargs = {
        "src": src,
        "tgt": tgt,
        "memory": memory,
        "tgt_mask": tgt_mask,
        "memory_mask": memory_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "tgt_key_padding_mask": tgt_key_padding_mask,
        "memory_key_padding_mask": memory_key_padding_mask,
    }
    if src_mask is not None:
        kwargs["src_mask"] = src_mask

    result = transformer(**kwargs)


    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        src = tf.constant(input_dict["src"])
        tgt = tf.constant(input_dict["tgt"])
        memory = tf.constant(input_dict["memory"])
        src_mask = tf.constant(input_dict["src_mask"]) if "src_mask" in input_dict else None
        tgt_mask = tf.constant(input_dict["tgt_mask"]) if "tgt_mask" in input_dict else None
        memory_mask = tf.constant(input_dict["memory_mask"]) if "memory_mask" in input_dict else None
        src_key_padding_mask = tf.constant(input_dict["src_key_padding_mask"]) if "src_key_padding_mask" in input_dict else None
        tgt_key_padding_mask = tf.constant(input_dict["tgt_key_padding_mask"]) if "tgt_key_padding_mask" in input_dict else None
        memory_key_padding_mask = tf.constant(input_dict["memory_key_padding_mask"]) if "memory_key_padding_mask" in input_dict else None

        d_model = input_dict["d_model"]
        nhead = input_dict["nhead"]
        num_encoder_layers = input_dict["num_encoder_layers"]
        num_decoder_layers = input_dict["num_decoder_layers"]
        dim_feedforward = input_dict.get("dim_feedforward", 2048)
        dropout = input_dict.get("dropout", 0.1)
        activation = input_dict.get("activation", "relu")
        custom_encoder = input_dict.get("custom_encoder", None)
        custom_decoder = input_dict.get("custom_decoder", None)
        layer_norm_eps = input_dict.get("layer_norm_eps", 1e-5)
        batch_first = input_dict.get("batch_first", False)
        norm_first = input_dict.get("norm_first", False)

        class FeedForward(tf.keras.layers.Layer):
            def __init__(self, dim_feedforward, d_model, dropout, activation):
                super(FeedForward, self).__init__()
                self.linear1 = tf.keras.layers.Dense(dim_feedforward)
                self.dropout = tf.keras.layers.Dropout(dropout)
                self.linear2 = tf.keras.layers.Dense(d_model)

                if activation == "relu":
                    self.activation = tf.nn.relu
                elif activation == "gelu":
                    self.activation = tf.nn.gelu
                else:
                    raise ValueError(f"Unsupported activation: {activation}")

            def call(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x

        class EncoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first):
                super(EncoderLayer, self).__init__()
                self.mha = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
                self.ffn = FeedForward(dim_feedforward, d_model, dropout, activation)
                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.dropout1 = tf.keras.layers.Dropout(dropout)
                self.dropout2 = tf.keras.layers.Dropout(dropout)
                self.norm_first = norm_first

            def call(self, x, mask=None):
                if self.norm_first:
                    x_norm = self.layernorm1(x)
                    attn_output = self.mha(x_norm, x_norm, x_norm, attention_mask=mask)
                    x = x + self.dropout1(attn_output)
                    x = self.layernorm2(x + self.ffn(self.layernorm1(x)))
                else:
                    attn_output = self.mha(x, x, x, attention_mask=mask)
                    x = self.layernorm1(x + self.dropout1(attn_output))
                    x = self.layernorm2(x + self.ffn(x))
                return x

        class DecoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first):
                super(DecoderLayer, self).__init__()
                self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
                self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
                self.ffn = FeedForward(dim_feedforward, d_model, dropout, activation)
                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
                self.dropout1 = tf.keras.layers.Dropout(dropout)
                self.dropout2 = tf.keras.layers.Dropout(dropout)
                self.dropout3 = tf.keras.layers.Dropout(dropout)
                self.norm_first = norm_first

            def call(self, x, memory, tgt_mask=None, memory_mask=None):
                if self.norm_first:
                    x_norm = self.layernorm1(x)
                    attn_output1 = self.mha1(x_norm, x_norm, x_norm, attention_mask=tgt_mask)
                    x = x + self.dropout1(attn_output1)
                    x_norm = self.layernorm2(x)
                    attn_output2 = self.mha2(x_norm, memory, memory, attention_mask=memory_mask)
                    x = x + self.dropout2(attn_output2)
                    x = self.layernorm3(x + self.ffn(self.layernorm3(x)))
                else:
                    attn_output1 = self.mha1(x, x, x, attention_mask=tgt_mask)
                    x = self.layernorm1(x + self.dropout1(attn_output1))
                    attn_output2 = self.mha2(x, memory, memory, attention_mask=memory_mask)
                    x = self.layernorm2(x + self.dropout2(attn_output2))
                    x = self.layernorm3(x + self.ffn(x))

                return x

        class Encoder(tf.keras.layers.Layer):
            def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first):
                super(Encoder, self).__init__()
                self.layers = [EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first) for _ in range(num_layers)]
                self.norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

            def call(self, x, mask=None):
                for layer in self.layers:
                    x = layer(x, mask=mask)
                x = self.norm(x)
                return x

        class Decoder(tf.keras.layers.Layer):
            def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first):
                super(Decoder, self).__init__()
                self.layers = [DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first) for _ in range(num_layers)]
                self.norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

            def call(self, x, memory, tgt_mask=None, memory_mask=None):
                for layer in self.layers:
                    x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
                x = self.norm(x)
                return x

        class Transformer(tf.keras.Model):
            def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first):
                super(Transformer, self).__init__()
                self.encoder = Encoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first)
                self.decoder = Decoder(num_decoder_layers, d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first)
                self.d_model = d_model
                self.nhead = nhead
                self.batch_first = batch_first

            def call(self, src, tgt, memory, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
                if src_mask is not None:
                    src_mask = tf.cast(src_mask, dtype=tf.float32)
                    src_mask = tf.where(tf.equal(src_mask, 0), -1e9, 0)
                else:
                    src_mask = None

                if tgt_mask is not None:
                    tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
                    tgt_mask = tf.where(tf.equal(tgt_mask, 0), -1e9, 0)

                if memory_mask is not None:
                    memory_mask = tf.cast(memory_mask, dtype=tf.float32)
                    memory_mask = tf.where(tf.equal(memory_mask, 0), -1e9, 0)
                else:
                    memory_mask = None
                    
                encoder_output = self.encoder(memory, mask=memory_mask)

                decoder_output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
                
                return decoder_output

        transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

        result = transformer(src, tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "src": np.random.rand(10, 32, 512).astype(np.float32),
        "tgt": np.random.rand(15, 32, 512).astype(np.float32),
        "memory": np.random.rand(20, 32, 512).astype(np.float32),
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "src_mask": np.random.rand(10,10).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()