import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


def _get_train_data(dataloader):
    (
        real_ind,
        real_seg,
        real_pos,
        real_img,
        cf_ind,
        cf_seg,
        cf_pos,
        cf_img,
        label,
        aug_label,
    ) = next(dataloader)
    max_seq_len = real_ind.get_shape().as_list()[-1]
    dim_img_feature = real_img.get_shape().as_list()[-3:]
    real_data = (
        tf.reshape(real_ind, shape=(-1, max_seq_len)),
        tf.reshape(real_seg, shape=(-1, max_seq_len)),
        tf.reshape(real_pos, shape=(-1, 1)),
        tf.reshape(real_img, shape=(-1, *dim_img_feature)),
    )
    cf_data = (
        tf.reshape(cf_ind, shape=(-1, max_seq_len)),
        tf.reshape(cf_seg, shape=(-1, max_seq_len)),
        tf.reshape(cf_pos, shape=(-1, 1)),
        tf.reshape(cf_img, shape=(-1, *dim_img_feature)),
    )
    label = tf.reshape(label, shape=(-1,))
    aug_label = tf.reshape(aug_label, shape=(-1,))
    return (real_data, cf_data, label, aug_label)


def _get_test_data(data):
    (
        real_ind,
        real_seg,
        real_pos,
        real_img,
        label,
    ) = data
    max_seq_len = real_ind.get_shape().as_list()[-1]
    dim_img_feature = real_img.get_shape().as_list()[-3:]
    real_data = (
        tf.reshape(real_ind, shape=(-1, max_seq_len)),
        tf.reshape(real_seg, shape=(-1, max_seq_len)),
        tf.reshape(real_pos, shape=(-1, 1)),
        tf.reshape(real_img, shape=(-1, *dim_img_feature)),
    )
    label = tf.reshape(label, shape=(-1,))
    return (real_data, label)


def _train_cf_model_with_batch(dataloader, model, optimizer, n_cf):
    real_data, cf_data, label, _ = _get_train_data(dataloader)
    with tf.GradientTape() as tape:
        real_logits, cr_logits, cf_prob = model(
            real_data, cf_data=cf_data, n_cf=n_cf, training=True
        )
        cf_label = tf.tile(tf.expand_dims(label, axis=-1), [1, n_cf])
        real_loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(label, real_logits)
        )
        cf_loss = tf.reduce_mean(
            tf.reduce_sum(
                cf_prob
                * tf.losses.sparse_categorical_crossentropy(cf_label, cr_logits),
                axis=-1,
            )
        )
        overall_loss = real_loss + cf_loss
    grads = tape.gradient(overall_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return overall_loss


def train_model(
    dataloader, model, optimizer, train_iter, cf: bool = False, n_cf: int = 1
):
    train_func = _train_cf_model_with_batch if cf else _train_model_with_batch
    train_params = [dataloader, model, optimizer]
    if cf:
        train_params.append(n_cf)

    losses = []
    loop = trange(1, train_iter + 1, ascii=True, ncols=80)
    for i in loop:
        loss = train_func(*train_params)
        losses.append(loss)
        loop.set_postfix_str("")
        loop.set_postfix_str("train_loss:{:^7.5f}".format(np.mean(losses)))


def evaluate_model(dataloader, model, dataset):
    labels = []
    preds = []
    for ind, seg, pos, img, label in tqdm(dataloader, ncols=80):
        data, label = _get_test_data([ind, seg, pos, img, label])
        logits = model(data, training=False)
        preds.append(logits)
        labels.append(label)
    preds = tf.argmax(tf.concat(preds, axis=0), axis=-1).numpy()
    labels = tf.concat(labels, axis=0).numpy()
    result = evaluate_mnet(labels, preds, dataset)
    return result


def evaluate_model_save_result(
    dataloader, model, dataset, prediction_path: str = "", feature_path: str = ""
):
    labels = []
    preds = []
    features = []
    for ind, seg, pos, img, label in tqdm(dataloader, ncols=80):
        data, label = _get_test_data([ind, seg, pos, img, label])
        if feature_path:
            logits, feature = model(data, training=False)
            features.append(feature)
        else:
            logits = model(data, training=False)
        preds.append(logits)
        labels.append(label)
    preds = tf.argmax(tf.concat(preds, axis=0), axis=-1).numpy()
    labels = tf.concat(labels, axis=0).numpy()
    result = evaluate_mnet(labels, preds, dataset)
    if prediction_path:
        pred_name = [dataset.LABEL[i] for i in preds]
        label_name = [dataset.LABEL[i] for i in labels]
        pd.to_pickle((pred_name, label_name), prediction_path)
    if feature_path:
        features = tf.concat(features, axis=0).numpy()
        pd.to_pickle(features, feature_path)
    return result


def evaluate_mnet(y_true, y_pred, dataset):
    acc = accuracy_score(y_true, y_pred)
    micro_p = precision_score(y_true, y_pred, average="micro")
    micro_r = recall_score(y_true, y_pred, average="micro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    result = {"acc": acc, "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1}
    return result


def _train_model_with_batch(dataloader, model, optimizer):
    real_data, _, label, _ = _get_train_data(dataloader)
    with tf.GradientTape() as tape:
        real_logits = model(real_data, training=True)
        real_loss = tf.reduce_mean(
            tf.losses.sparse_categorical_crossentropy(label, real_logits)
        )
        overall_loss = real_loss
    grads = tape.gradient(overall_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return overall_loss
