import tensorflow as tf
from vit_keras import vit
import tensorflow.keras.backend as K
from transformer import EncoderLayer
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from keras_bert import load_trained_model_from_checkpoint


class BERTEmbedding(tf.keras.models.Model):
    def __init__(self, bert_path: str, fine_tune: bool = False):
        super(BERTEmbedding, self).__init__()
        self.__ckpt_path = f"{bert_path}/bert_model.ckpt"
        self.__config_path = f"{bert_path}/bert_config.json"
        self.bert_model = load_trained_model_from_checkpoint(
            self.__config_path, self.__ckpt_path, seq_len=None
        )
        for l in self.bert_model.layers:
            l.trainable = fine_tune

    @tf.function
    def call(self, ind, seg):
        return self.bert_model([ind, seg])


class MKGformer(tf.keras.models.Model):
    def __init__(
        self, sentence_encoder, n_label, hidden_size: int = 768, use_cf: bool = False
    ):
        super(MKGformer, self).__init__()
        self.use_cf = use_cf
        self.hidden_size = hidden_size
        self.n_label = n_label
        self.sentence_encoder = sentence_encoder
        self.image_encoder = vit.vit_b16(
            image_size=(384, 384),
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        for l in self.image_encoder.layers:
            l.trainable = False
        self.cross_transformer1 = EncoderLayer(hidden_size, 12, hidden_size)
        self.cross_transformer2 = EncoderLayer(hidden_size, 12, hidden_size)
        self.ffn = Sequential(
            [Dense(hidden_size, activation="relu"), Dense(hidden_size)]
        )
        self.fc = Dense(n_label)
        self.w_v = Dense(hidden_size)
        self.w_t = Dense(hidden_size)
        self.cf_gate = Dense(1, activation="sigmoid")
        self.dropout = Dropout(0.1)

    @tf.function
    def PGI(self, h_t, h_v):
        k_t = self.w_t(h_t)
        k_v = self.w_v(h_v)
        h_tv = tf.concat([k_t, k_v], axis=1)
        h_mm = self.cross_transformer2(h_tv, h_tv, h_v)
        return h_mm

    @tf.function
    def CAF(self, h_t, h_v):
        S = tf.nn.softmax(h_t @ tf.transpose(h_v, perm=[0, 2, 1]), axis=-1)
        agg = S @ h_v
        h_mm = self.ffn(tf.concat([h_t, agg], axis=-1))
        return h_mm

    @tf.function
    def get_tensor_by_index(self, tensor, bs, pos):
        indices = tf.concat(
            [tf.expand_dims(tf.range(bs, dtype=tf.int64), axis=1), pos],
            axis=-1,
        )
        _tensor = tf.gather_nd(tensor, indices)
        return _tensor

    @tf.function
    def mm_fusion(self, data, training, stop_gd: bool = False):
        ind, seg, pos, img = data
        h_t = self.sentence_encoder(ind, seg)
        if stop_gd:
            h_t = tf.stop_gradient(h_t)
        h_v = self.image_encoder(img)
        h_v = tf.zeros_like(h_v)
        h_t_p = self.cross_transformer1(h_t, h_t, h_t)
        h_v = self.PGI(h_t, h_v)
        h_mm = self.CAF(h_t_p, h_v)

        batch_size = tf.shape(ind)[0]
        h_pos = self.get_tensor_by_index(h_mm, batch_size, pos)
        h_cls = h_mm[:, 0, :]
        h_e = tf.concat([h_pos, h_cls], axis=-1)
        if training:
            h_e = self.dropout(h_e, training=training)
        logits = self.fc(h_e)
        return h_e, logits

    @tf.function
    def call(self, real_data, cf_data=None, n_cf=1, training=False):
        h_e_r, h_r_logits = self.mm_fusion(real_data, training)
        h_r_prob = tf.nn.softmax(h_r_logits, axis=-1)
        if self.use_cf and training:
            h_e_c, h_c_logits = self.mm_fusion(cf_data, training, stop_gd=True)
            hidden_size = tf.shape(h_e_c)[-1]
            h_e_r = tf.tile(tf.expand_dims(h_e_r, axis=1), [1, n_cf, 1])
            h_e_c = tf.reshape(h_e_c, shape=(-1, n_cf, hidden_size))
            cf_prob = tf.nn.softmax(
                tf.squeeze(self.cf_gate(h_e_c), axis=-1),
                axis=-1,
            )

            h_c_logits = tf.reshape(h_c_logits, shape=(-1, n_cf, self.n_label))
            h_r_logits = tf.expand_dims(h_r_logits, axis=1)
            h_c_prob = tf.nn.softmax(h_r_logits - h_c_logits, axis=-1)
            return h_r_prob, h_c_prob, cf_prob
        return h_r_prob


class UMT(tf.keras.models.Model):
    def __init__(
        self,
        sentence_encoder,
        n_label,
        hidden_size: int = 768,
        use_cf: bool = False,
    ):
        super(UMT, self).__init__()
        self.use_cf = use_cf
        self.hidden_size = hidden_size
        self.n_label = n_label
        self.sentence_encoder = sentence_encoder
        self.image_encoder = vit.vit_b16(
            image_size=(384, 384),
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        for l in self.image_encoder.layers:
            l.trainable = False
        self.cross_transformer1 = EncoderLayer(hidden_size, 12, hidden_size)
        self.cross_transformer2 = EncoderLayer(hidden_size, 12, hidden_size)
        self.cross_transformer3 = EncoderLayer(hidden_size, 12, hidden_size)
        self.gate = Dense(hidden_size, activation="sigmoid")
        self.fc = Dense(n_label)
        self.cf_gate = Dense(1, activation="sigmoid")
        self.dropout = Dropout(0.1)

    @tf.function
    def get_tensor_by_index(self, tensor, bs, pos):
        indices = tf.concat(
            [tf.expand_dims(tf.range(bs, dtype=tf.int64), axis=1), pos],
            axis=-1,
        )
        _tensor = tf.gather_nd(tensor, indices)
        return _tensor

    @tf.function
    def mm_fusion(self, data, training, stop_gd: bool = False):
        ind, seg, pos, img = data
        h_t = self.sentence_encoder(ind, seg)
        if stop_gd:
            h_t = tf.stop_gradient(h_t)
        h_v = self.image_encoder(img)
        h_v = tf.zeros_like(h_v)
        h_1 = self.cross_transformer1(h_t, h_t, h_v)
        h_2 = self.cross_transformer2(h_v, h_v, h_t)
        h_3 = self.cross_transformer3(h_1, h_1, h_t)

        g = self.gate(tf.concat([h_3, h_2], axis=-1))
        h_mm = tf.concat([h_3, h_2 * g], axis=-1)

        batch_size = tf.shape(ind)[0]
        h_pos = self.get_tensor_by_index(h_mm, batch_size, pos)
        # h_e = tf.concat([h_ph, h_pt], axis=-1)
        h_cls = h_mm[:, 0, :]
        h_e = tf.concat([h_pos, h_cls], axis=-1)
        if training:
            h_e = self.dropout(h_e, training=training)
        logits = self.fc(h_e)
        return h_e, logits

    @tf.function
    def call(self, real_data, cf_data=None, n_cf=1, training=False):
        h_e_r, h_r_logits = self.mm_fusion(real_data, training)
        h_r_prob = tf.nn.softmax(h_r_logits, axis=-1)
        if self.use_cf and training:
            h_e_c, h_c_logits = self.mm_fusion(cf_data, training, stop_gd=True)
            hidden_size = tf.shape(h_e_c)[-1]
            h_e_r = tf.tile(tf.expand_dims(h_e_r, axis=1), [1, n_cf, 1])
            h_e_c = tf.reshape(h_e_c, shape=(-1, n_cf, hidden_size))
            cf_prob = tf.nn.softmax(
                tf.squeeze(self.cf_gate(h_e_c), axis=-1),
                axis=-1,
            )

            h_c_logits = tf.reshape(h_c_logits, shape=(-1, n_cf, self.n_label))
            h_r_logits = tf.expand_dims(h_r_logits, axis=1)
            h_c_prob = tf.nn.softmax(h_r_logits - h_c_logits, axis=-1)
            return h_r_prob, h_c_prob, cf_prob
        return h_r_prob
