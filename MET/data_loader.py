import os
import re
import codecs
import json
import hashlib
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from threading import Thread
from vit_keras import vit
from typing import List, Dict
from tensorflow.data import Dataset
from keras_bert.tokenizer import Tokenizer
from tensorflow.keras.preprocessing import image
from keras_bert import TOKEN_CLS, TOKEN_MASK, TOKEN_SEP
from tensorflow.keras.applications.resnet import preprocess_input


class Entity(object):
    """
    Named Entity
    """

    def __init__(self, name, type, h_pos, t_pos, url):
        self.__name = name
        self.__type = type
        self.__h_pos = h_pos
        self.__t_pos = t_pos
        self.__url = url

    @property
    def name(self):
        return self.__name

    @property
    def type(self):
        return self.__type

    @property
    def pos(self):
        return [self.__h_pos, self.__t_pos]

    @property
    def url(self):
        return self.__url

    def __str__(self) -> str:
        return str({"name": self.name, "type": self.type, "pos": self.pos})


class CopyMNetSample(object):
    def __init__(self, sentence, image, topic, entity):
        self.__sentence = sentence
        self.__image = image
        self.__topic = topic
        self.__entity = entity

    @property
    def topic(self):
        return self.__topic

    @property
    def image(self):
        return self.__image

    @property
    def sentence(self):
        return self.__sentence

    @property
    def entity(self):
        return self.__entity


class MNetSample(Thread):
    """
    Multimodal Named Entity Typing Sample
    """

    def __init__(self, **kwargs) -> None:
        Thread.__init__(self)
        if "sample" not in kwargs:
            self.__sentence = kwargs["sentence"]
            self.__img_url = kwargs["img_url"]
            self.__data_path = kwargs["data_path"]
            self.__topic = kwargs["topic"]
            self.__entities = [
                Entity(name, type, h_pos, t_pos, url)
                for name, type, h_pos, t_pos, url in kwargs["entity"]
            ]
        else:
            sample = kwargs["sample"]
            self.__sentence = sample.sentence
            self.__image = sample.image
            self.__topic = sample.topic
            self.__entities = kwargs["entity"]

    def run(self):
        file_name = self.__parse_img_file_name(self.__img_url)
        self.__image = self.__load_image(f"{self.__data_path}/wikinewsImgs/{file_name}")

    @property
    def sentence(self):
        return self.__sentence

    @property
    def image(self):
        return self.__image

    @property
    def topic(self):
        return self.__topic

    @property
    def entity(self):
        return self.__entities

    def __str__(self) -> str:
        entities = "".join([str(e) for e in self.entity])
        return str({"sentence": self.sentence, "topic": self.topic, "entity": entities})

    def __parse_img_file_name(self, url: str):
        m_img = url.split("/")[-1]
        prefix = hashlib.md5(m_img.encode()).hexdigest()
        suffix = re.sub(
            r"(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))", "", m_img
        )
        m_img = prefix + suffix
        m_img = m_img.replace(".svg", ".png").replace(".SVG", ".png")
        return m_img

    def __load_image(self, fname: str, image_size: int = 384):
        if not os.path.exists(fname):
            fname = f"{self.__data_path}/wikinewsImgs/17_06_4705.jpg"
        try:
            img = image.load_img(fname, target_size=(image_size, image_size))
        except Exception:
            fname = f"{self.__data_path}/wikinewsImgs/17_06_4705.jpg"
            img = image.load_img(fname, target_size=(image_size, image_size))
        x = image.img_to_array(img)
        # x = preprocess_input(x)
        x = vit.preprocess_inputs(x).reshape(image_size, image_size, 3)
        return x


class MNetDataset(object):
    def __init__(self, data_path: str, n_train: int = 100):
        self.__n_train = n_train
        self.__labels = []
        self.__samples = self.__collect_data(data_path)

    @property
    def LABEL(self):
        return self.__labels

    def label2id(self, label):
        return self.__labels.index(label)

    def Data(self, dtype: str):
        return self.__samples[dtype]

    def __collect_data(self, data_path: str) -> List[MNetSample]:
        samples = {}
        for dtype in ["train", "valid", "test"]:
            tmp_samples = []
            with open(f"{data_path}/{dtype}.json", "r") as fr:
                json_data = json.load(fr)
                for sentence, image_url, topic, entities in tqdm(
                    json_data, ascii=True, ncols=80
                ):
                    sample = MNetSample(
                        sentence=sentence,
                        img_url=image_url,
                        topic=topic,
                        entity=entities,
                        data_path=data_path,
                    )
                    for e in sample.entity:
                        if e.type not in self.__labels:
                            self.__labels.append(e.type)
                    sample.start()
                    tmp_samples.append(sample)
                for sample in tmp_samples:
                    sample.join()
            samples[dtype] = []
            for sample in tmp_samples:
                for e in sample.entity:
                    samples[dtype].append(MNetSample(sample=sample, entity=e))

        if self.__n_train > 0:
            label_percentage = {k: 0 for k in self.__labels}
            for s in samples["train"]:
                label_percentage[s.entity.type] += 1
            for k, v in label_percentage.items():
                label_percentage[k] = 1.0 - v / len(samples["train"])
            p = [label_percentage[s.entity.type] for s in samples["train"]]
            train_idx = np.arange(len(samples["train"]))
            selected_train_idx = np.random.choice(
                train_idx, size=self.__n_train, replace=False, p=p / np.sum(p)
            )
            selected_train_data = [samples["train"][i] for i in selected_train_idx]
            samples["train"] = selected_train_data
        return samples


class TextProcessor(object):
    def __init__(self, bert_path: str, max_length: int = 128):
        self.__max_length = max_length
        self.__tokenizer = self.__load_tokenizer(bert_path)

    @property
    def VOCAB_SIZE(self):
        return len(self.__tokenizer._token_dict)

    def encode(self, sentence: str, max_seq_len: int):
        indices, segments = self.__tokenizer.encode(first=sentence, max_len=max_seq_len)
        return indices, segments

    def tokenize(self, sentence: str, pos: List):
        h_in_idx, t_in_idx = pos
        context1 = sentence[0:h_in_idx]
        entity = sentence[h_in_idx:t_in_idx]
        context2 = sentence[t_in_idx:]

        tokens_context1 = self.__tokenizer.tokenize(context1)[1:-1]
        tokens_entity = (
            ["[unused0]"] + self.__tokenizer.tokenize(entity)[1:-1] + ["[unused1]"]
        )
        tokens_context2 = self.__tokenizer.tokenize(context2)[1:-1]

        tokens = (
            [TOKEN_CLS]
            + tokens_context1
            + tokens_entity
            + tokens_context2
            + [TOKEN_SEP]
        )
        pos_in_index = tokens.index("[unused0]")

        indices = self.__tokenizer._convert_tokens_to_ids(tokens)
        while len(indices) < self.__max_length:
            indices.append(0)
        indices = indices[: self.__max_length]
        segments = np.zeros_like(indices).tolist()

        return (indices, segments, pos_in_index)

    def __load_tokenizer(self, bert_path):
        token_dict = {}
        with codecs.open(f"{bert_path}/vocab.txt", "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return Tokenizer(token_dict)


class TrainDataGenerator(object):
    def __init__(
        self,
        dataset: MNetDataset,
        textProcessor: TextProcessor,
        ex_img: bool,
        ex_ent: bool,
        ex_cxt: bool,
    ):
        self.__textProcessor = textProcessor
        self.__dataset = dataset
        self.__samples = dataset.Data("train")
        self.__sample_idx = np.arange(len(self.__samples))
        self.__ex_img = ex_img
        self.__ex_ent = ex_ent
        self.__ex_cxt = ex_cxt

    def __exchange_context(self, i, j):
        sample1 = self.__samples[i]
        sample2 = self.__samples[j]

        sample1_sentence = sample1.sentence
        sample1_ent = sample1.entity
        sample1_ent_sentence = sample1_sentence[sample1_ent.pos[0] : sample1_ent.pos[1]]

        sample2_sentence = sample2.sentence
        sample2_ent_pos = sample2.entity.pos
        context1 = sample2_sentence[: sample2_ent_pos[0]]
        context2 = sample2_sentence[sample2_ent_pos[1] :]

        start_index = len(context1)
        end_index = start_index + len(sample1_ent_sentence)
        counterfactual_entity = Entity(
            sample1_ent.name, sample1_ent.type, start_index, end_index, sample1_ent.url
        )
        counterfactual_sentence = context1 + sample1_ent_sentence + context2
        counterfactual_sample = CopyMNetSample(
            counterfactual_sentence, sample1.image, sample1.topic, counterfactual_entity
        )
        return counterfactual_sample

    def __exchange_entity(self, i, j):
        sample1 = self.__samples[i]
        sample2 = self.__samples[j]

        sample1_sentence = sample1.sentence
        sample1_ent = sample1.entity
        context1 = sample1_sentence[: sample1_ent.pos[0]]
        context2 = sample1_sentence[sample1_ent.pos[1] :]

        sample2_sentence = sample2.sentence
        sample2_ent = sample2.entity
        sample2_ent_sentence = sample2_sentence[sample2_ent.pos[0] : sample2_ent.pos[1]]

        start_index = len(context1)
        end_index = start_index + len(sample2_ent_sentence)
        counterfactual_entity = Entity(
            sample2_ent.name, sample1_ent.type, start_index, end_index, sample2_ent.url
        )
        counterfactual_sentence = context1 + sample2_ent_sentence + context2
        counterfactual_sample = CopyMNetSample(
            counterfactual_sentence, sample1.image, sample1.topic, counterfactual_entity
        )
        return counterfactual_sample

    def __exchange_image(self, i, j):
        sample1 = self.__samples[i]
        sample2 = self.__samples[j]

        sample1_sentence = sample1.sentence
        sample1_ent = sample1.entity
        sample2_image = sample2.image

        counterfactual_sample = CopyMNetSample(
            sample1_sentence, sample2_image, sample1.topic, sample1_ent
        )
        return counterfactual_sample

    def __convert_data(self, samples):
        parsed_data = {"indices": [], "segments": [], "pos": [], "image": []}
        for sample in samples:
            indices, segments, pos = self.__textProcessor.tokenize(
                sample.sentence, sample.entity.pos
            )
            parsed_data["indices"].append(indices)
            parsed_data["segments"].append(segments)
            parsed_data["pos"].append(pos)
            parsed_data["image"].append(sample.image)
        return parsed_data

    def _generate_data(self):
        while True:
            real_set, counterfactual_set, label, aug_label = self.__get_item()
            yield real_set["indices"], real_set["segments"], real_set["pos"], real_set[
                "image"
            ], counterfactual_set["indices"], counterfactual_set[
                "segments"
            ], counterfactual_set[
                "pos"
            ], counterfactual_set[
                "image"
            ], label, aug_label

    def __get_item(self):
        i, j = np.random.choice(self.__sample_idx, size=2, replace=False).tolist()
        sample = self.__samples[i]
        counterfactual_samples = []
        counterfactual_labels = []
        if self.__ex_img:
            counterfactual_sample = self.__exchange_image(i, j)
            counterfactual_samples.append(counterfactual_sample)
            counterfactual_labels.append(sample.entity.type)
        if self.__ex_ent:
            counterfactual_sample = self.__exchange_entity(i, j)
            counterfactual_samples.append(counterfactual_sample)
            counterfactual_labels.append(self.__samples[j].entity.type)
        if self.__ex_cxt:
            counterfactual_sample = self.__exchange_context(i, j)
            counterfactual_samples.append(counterfactual_sample)
            counterfactual_labels.append(self.__samples[j].entity.type)

        real_set = self.__convert_data([sample])
        counterfactual_set = self.__convert_data(counterfactual_samples)
        label = [self.__dataset.label2id(sample.entity.type)]
        aug_label = [self.__dataset.label2id(t) for t in counterfactual_labels]
        return (real_set, counterfactual_set, label, aug_label)


def get_train_dataloader(batch_size, dataset, textProcessor, ex_img, ex_ent, ex_cxt):
    generator = TrainDataGenerator(dataset, textProcessor, ex_img, ex_ent, ex_cxt)
    dataloader = Dataset.from_generator(
        generator._generate_data,
        tuple([tf.int64, tf.int64, tf.int64, tf.float32] * 2 + [tf.int64] * 2),
        tuple(
            [
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None, None, None]),
            ]
            * 2
            + [tf.TensorShape([None])] * 2
        ),
    )
    dataloader = dataloader.batch(batch_size)
    return iter(dataloader)


class ValidTestDataloader(object):
    def __init__(
        self, dataset: MNetDataset, textProcessor: TextProcessor, batch_size: int = 8
    ):
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__textProcessor = textProcessor
        self._valid_dataloader = self.__convert_data(dataset.Data("valid"))
        self._test_dataloader = self.__convert_data(dataset.Data("test"))

    def Data(self, dtype: str):
        return getattr(self, f"_{dtype}_dataloader")

    def __convert_data(self, samples):
        parsed_data = {
            "indices": [],
            "segments": [],
            "pos": [],
            "image": [],
            "label": [],
        }
        for sample in samples:
            indices, segments, pos = self.__textProcessor.tokenize(
                sample.sentence, sample.entity.pos
            )
            parsed_data["indices"].append(indices)
            parsed_data["segments"].append(segments)
            parsed_data["pos"].append(pos)
            parsed_data["image"].append(sample.image)
            parsed_data["label"].append(self.__dataset.label2id(sample.entity.type))
        dataloader = Dataset.from_tensor_slices(
            tuple([np.array(v) for v in parsed_data.values()])
        ).batch(self.__batch_size)
        return dataloader


if __name__ == "__main__":
    dataset = MNetDataset("./dataset")
    textProcessor = TextProcessor("../../ZS-MNET/pretrain/cased_L-12_H-768_A-12")
    train_loader = get_train_dataloader(2, dataset, textProcessor, 1, 1, 1)
    valid_test_dataloader = ValidTestDataloader(dataset, textProcessor, batch_size=2)

    for (
        real_ind,
        real_seg,
        real_p,
        real_img,
        cf_ind,
        cf_seg,
        cf_p,
        cf_img,
        label,
    ) in train_loader:
        print(real_ind)
        print(real_seg)
        print(real_p)
        print(cf_ind)
        print(cf_seg)
        print(cf_p)
        print(cf_img)
        print(label)
        break

    for ind, seg, p, img, label in valid_test_dataloader.Data("valid"):
        print(ind)
        print(seg)
        print(p)
        print(img)
        print(label)
        break

    for ind, seg, p, img, label in valid_test_dataloader.Data("test"):
        print(ind)
        print(seg)
        print(p)
        print(img)
        print(label)
        break
