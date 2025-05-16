import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    tgt = torch.tensor(input_dict["tgt"])
    memory = torch.tensor(input_dict["memory"])
    tgt_mask = torch.tensor(input_dict["tgt_mask"]) if "tgt_mask" in input_dict else None
    memory_mask = torch.tensor(input_dict["memory_mask"]) if "memory_mask" in input_dict else None
    tgt_key_padding_mask = torch.tensor(input_dict["tgt_key_padding_mask"]) if "tgt_key_padding_mask" in input_dict else None
    memory_key_padding_mask = torch.tensor(input_dict["memory_key_padding_mask"]) if "memory_key_padding_mask" in input_dict else None
    
    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=input_dict["d_model"], nhead=input_dict["nhead"], dim_feedforward=input_dict.get("dim_feedforward", 2048), dropout=input_dict.get("dropout", 0.1), activation=input_dict.get("activation", "relu"), layer_norm_eps=input_dict.get("layer_norm_eps", 1e-5), batch_first=input_dict.get("batch_first", False), norm_first=input_dict.get("norm_first", False))
    num_layers = input_dict["num_layers"]
    norm = torch.nn.LayerNorm(input_dict["d_model"]) if input_dict.get("norm", True) else None
    decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=norm)
    
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
        decoder = decoder.cuda()
        
    result = decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

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
        tgt = tf.constant(input_dict["tgt"], dtype=tf.float32)
        memory = tf.constant(input_dict["memory"], dtype=tf.float32)
        tgt_mask = tf.constant(input_dict["tgt_mask"], dtype=tf.float32) if "tgt_mask" in input_dict else None
        memory_mask = tf.constant(input_dict["memory_mask"], dtype=tf.float32) if "memory_mask" in input_dict else None
        tgt_key_padding_mask = tf.constant(input_dict["tgt_key_padding_mask"], dtype=tf.float32) if "tgt_key_padding_mask" in input_dict else None
        memory_key_padding_mask = tf.constant(input_dict["memory_key_padding_mask"], dtype=tf.float32) if "memory_key_padding_mask" in input_dict else None
        
        d_model = input_dict["d_model"]
        nhead = input_dict["nhead"]
        dim_feedforward = input_dict.get("dim_feedforward", 2048)
        dropout = input_dict.get("dropout", 0.1)
        num_layers = input_dict["num_layers"]
        
        def scaled_dot_product_attention(q, k, v, mask=None):
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)
                
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, v)
            return output, attention_weights
        
        def split_heads(x, d_model, num_heads):
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, (batch_size, -1, num_heads, d_model // num_heads))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        
        def combine_heads(x, d_model, num_heads):
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[2]
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            return tf.reshape(x, (batch_size, seq_len, d_model))

        def point_wise_feed_forward_network(d_model, dff):
            return tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation='relu'),
                tf.keras.layers.Dense(d_model)
            ])

        class DecoderLayer(tf.keras.layers.Layer):
            def __init__(self, d_model, num_heads, dff, rate=0.1):
                super(DecoderLayer, self).__init__()
                
                self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
                self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
                
                self.ffn = point_wise_feed_forward_network(d_model, dff)
                
                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                
                self.dropout1 = tf.keras.layers.Dropout(rate)
                self.dropout2 = tf.keras.layers.Dropout(rate)
                self.dropout3 = tf.keras.layers.Dropout(rate)
                
            def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
                attn1, _ = self.mha1(x, x, x, attention_mask=look_ahead_mask, return_attention_scores=False)  # Self attention (q=x, k=x, v=x)
                attn1 = self.dropout1(attn1, training=training)
                out1 = self.layernorm1(attn1 + x)
                
                attn2, _ = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask, return_attention_scores=False)  # Encoder-decoder attention (q=out1, k=enc_output, v=enc_output)
                attn2 = self.dropout2(attn2, training=training)
                out2 = self.layernorm2(attn2 + out1)
                
                ffn_output = self.ffn(out2)
                ffn_output = self.dropout3(ffn_output, training=training)
                out3 = self.layernorm3(ffn_output + out2)
                
                return out3
        
        class Decoder(tf.keras.layers.Layer):
            def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
                super(Decoder, self).__init__()
                
                self.num_layers = num_layers
                self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
                self.dropout = tf.keras.layers.Dropout(rate)
            
            def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
                seq_len = tf.shape(x)[1]
                attention_weights = {}
                
                x = self.dropout(x, training=training)
                
                for i in range(self.num_layers):
                    x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
                    
                return x, attention_weights

        look_ahead_mask = None
        if tgt_mask is not None:
          look_ahead_mask = tgt_mask
        
        if look_ahead_mask is not None:
            look_ahead_mask = tf.cast(look_ahead_mask, dtype=tf.float32)
            look_ahead_mask = tf.where(look_ahead_mask < 0.5, -1e9, 0.0)

        padding_mask = None
        if memory_mask is not None:
          padding_mask = memory_mask

        if padding_mask is not None:
            padding_mask = tf.cast(padding_mask, dtype=tf.float32)
            padding_mask = tf.where(padding_mask < 0.5, -1e9, 0.0)
        
        decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=nhead, dff=dim_feedforward, rate=dropout)
        
        result, _ = decoder(tgt, memory, training=False, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        
        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    d_model = 512
    nhead = 8
    num_layers = 2
    
    tgt_len = 10
    memory_len = 20
    batch_size = 2

    input_data = {
        "tgt": np.random.rand(batch_size, tgt_len, d_model).astype(np.float32),
        "memory": np.random.rand(batch_size, memory_len, d_model).astype(np.float32),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "tgt_mask": np.random.rand(batch_size, tgt_len, tgt_len).astype(np.float32) < 0.5,
        "memory_mask": np.random.rand(batch_size, tgt_len, memory_len).astype(np.float32) < 0.5,
        "tgt_key_padding_mask": np.random.rand(batch_size, tgt_len).astype(np.float32) < 0.5,
        "memory_key_padding_mask": np.random.rand(batch_size, memory_len).astype(np.float32) < 0.5,
        "dim_feedforward": 2048,
        "dropout": 0.1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()