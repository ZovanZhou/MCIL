import copy
import random
import codecs
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from vit_keras import vit
from threading import Thread
from tensorflow.data import Dataset
from typing import Dict, List, Tuple
from keras_bert.tokenizer import Tokenizer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input


class TextProcessor(object):
    def __init__(
        self, bert_path: str, max_length: int = 128, mask_entity: bool = False
    ):
        self.__mask_entity = mask_entity
        self.__max_length = max_length
        self.__tokenizer = self.__load_tokenizer(bert_path)

    def tokenize_sentence(self, sentence: str, max_len=16):
        indices, segments = self.__tokenizer.encode(sentence, max_len=max_len)
        return (indices, segments)

    def tokenize_sample(self, raw_tokens: List, pos_head: List, pos_tail: List):
        pos_head[-1] -= 1
        pos_tail[-1] -= 1
        tokens = ["[CLS]"]
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append("[unused0]")
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append("[unused1]")
                pos2_in_index = len(tokens)
            if self.__mask_entity and (
                (pos_head[0] <= cur_pos and cur_pos <= pos_head[-1])
                or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])
            ):
                tokens += ["[unused4]"]
            else:
                tokens += self.__tokenizer.tokenize(token)[1:-1]
            if cur_pos == pos_head[-1]:
                tokens.append("[unused2]")
            if cur_pos == pos_tail[-1]:
                tokens.append("[unused3]")
            cur_pos += 1
        tokens.append("[SEP]")
        indices = self.__tokenizer._convert_tokens_to_ids(tokens)

        # padding
        while len(indices) < self.__max_length:
            indices.append(0)
        indices = indices[: self.__max_length]
        segments = np.zeros_like(indices).tolist()

        pos1_in_index = min(self.__max_length, pos1_in_index)
        pos2_in_index = min(self.__max_length, pos2_in_index)

        return indices, segments, pos1_in_index - 1, pos2_in_index - 1

    def __load_tokenizer(self, bert_path):
        token_dict = {}
        with codecs.open(f"{bert_path}/vocab.txt", "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return Tokenizer(token_dict)


class MRelSample(Thread):
    def __init__(self, dict_data: Dict, image_data_path: str):
        Thread.__init__(self)
        self.__token = dict_data["token"]
        self.__head_ent = dict_data["h"]
        self.__tail_ent = dict_data["t"]
        self.__image_data_path = image_data_path
        self.__img_id = dict_data["img_id"]
        self.__relation = dict_data["relation"]

    def run(self):
        self.__image = self.__load_image(f"{self.__image_data_path}/{self.__img_id}")

    def __load_image(self, fname: str, image_size: int = 384):
        img = image.load_img(fname, target_size=(image_size, image_size))
        x = image.img_to_array(img)
        # x = preprocess_input(x)
        x = vit.preprocess_inputs(x).reshape(image_size, image_size, 3)
        return x

    @property
    def image_id(self):
        return self.__img_id

    @property
    def image(self):
        return self.__image

    @property
    def token(self):
        return self.__token

    @property
    def head_entity(self):
        return self.__head_ent

    @property
    def tail_entity(self):
        return self.__tail_ent

    @property
    def relation(self):
        return self.__relation

    def __str__(self):
        return str(
            {
                "token": self.__token,
                "head_entity": self.__head_ent,
                "tail_entity": self.__tail_ent,
                "relation": self.__relation,
            }
        )


class CopyMRelSample(MRelSample):
    def __init__(self, token, head_ent, tail_ent, image):
        self.__token = token
        self.__head_ent = head_ent
        self.__tail_ent = tail_ent
        self.__image = image

    @property
    def image(self):
        return self.__image

    @property
    def token(self):
        return self.__token

    @property
    def head_entity(self):
        return self.__head_ent

    @property
    def tail_entity(self):
        return self.__tail_ent

    @property
    def relation(self):
        return self.__relation


class MRelDataset(object):
    def __init__(self, data_path: str, n_train: int = 100):
        self.__n_train = n_train
        self.__relations = []
        self.__samples = self.__collect_data(data_path)

    @property
    def RELATION(self):
        return self.__relations

    def relation2id(self, relation):
        return self.__relations.index(relation)

    def Data(self, dtype: str):
        return self.__samples[dtype]

    def __collect_data(self, data_path: str) -> List[MRelSample]:
        samples = {}
        for dtype in ["train", "val", "test"]:
            samples[dtype] = []
            with open(f"{data_path}/{dtype}.txt", "r") as fr:
                for line in tqdm(fr.readlines(), ncols=80, ascii=True):
                    sample = MRelSample(eval(line), f"{data_path}/image")
                    if sample.relation not in self.__relations:
                        self.__relations.append(sample.relation)
                    sample.start()
                    samples[dtype].append(sample)
                for sample in samples[dtype]:
                    sample.join()

        if self.__n_train > 0:
            relation_percentage = {k: 0 for k in self.__relations}
            for s in samples["train"]:
                relation_percentage[s.relation] += 1
            for k, v in relation_percentage.items():
                relation_percentage[k] = 1.0 - v / len(samples["train"])
            p = [relation_percentage[s.relation] for s in samples["train"]]
            train_idx = np.arange(len(samples["train"]))
            selected_train_idx = np.random.choice(
                train_idx, size=self.__n_train, replace=False, p=p / np.sum(p)
            )
            selected_train_data = [samples["train"][i] for i in selected_train_idx]
            samples["train"] = selected_train_data
        return samples


class TrainDataGenerator(object):
    def __init__(
        self,
        dataset: MRelDataset,
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

        sample1_token = sample1.token
        sample1_head_ent = sample1.head_entity
        sample1_head_ent_token = sample1_token[
            sample1_head_ent["pos"][0] : sample1_head_ent["pos"][1]
        ]
        sample1_tail_ent = sample1.tail_entity
        sample1_tail_ent_token = sample1_token[
            sample1_tail_ent["pos"][0] : sample1_tail_ent["pos"][1]
        ]

        sample2_token = sample2.token
        sample2_head_ent_pos = sample2.head_entity["pos"]
        sample2_tail_ent_pos = sample2.tail_entity["pos"]

        if sample2_head_ent_pos[1] - 1 < sample2_tail_ent_pos[0]:
            context1 = sample2_token[: sample2_head_ent_pos[0]]
            context2 = sample2_token[sample2_head_ent_pos[1] : sample2_tail_ent_pos[0]]
            context3 = sample2_token[sample2_tail_ent_pos[1] :]
            counterfactual_token = (
                context1
                + sample1_head_ent_token
                + context2
                + sample1_tail_ent_token
                + context3
            )
            start_index = len(context1)
            counterfactual_head_ent = {
                "pos": [start_index, start_index + len(sample1_head_ent_token)],
                "name": sample1_head_ent["name"],
            }
            start_index = len(context1) + len(sample1_head_ent_token) + len(context2)
            counterfactual_tail_ent = {
                "pos": [start_index, start_index + len(sample1_tail_ent_token)],
                "name": sample1_tail_ent["name"],
            }
            counterfactual_sample = CopyMRelSample(
                counterfactual_token,
                counterfactual_head_ent,
                counterfactual_tail_ent,
                sample1.image,
            )
            return counterfactual_sample
        elif sample2_head_ent_pos[0] > sample2_tail_ent_pos[1] - 1:
            context1 = sample2_token[: sample2_tail_ent_pos[0]]
            context2 = sample2_token[sample2_tail_ent_pos[1] : sample2_head_ent_pos[0]]
            context3 = sample2_token[sample2_head_ent_pos[1] :]
            counterfactual_token = (
                context1
                + sample1_tail_ent_token
                + context2
                + sample1_head_ent_token
                + context3
            )
            start_index = len(context1)
            counterfactual_tail_ent = {
                "pos": [start_index, start_index + len(sample1_tail_ent_token)],
                "name": sample1_tail_ent["name"],
            }
            start_index = len(context1) + len(sample1_tail_ent_token) + len(context2)
            counterfactual_head_ent = {
                "pos": [start_index, start_index + len(sample1_head_ent_token)],
                "name": sample1_head_ent["name"],
            }
            counterfactual_sample = CopyMRelSample(
                counterfactual_token,
                counterfactual_head_ent,
                counterfactual_tail_ent,
                sample1.image,
            )
            return counterfactual_sample
        else:
            return self.__exchange_image(i, j)

    def __exchange_entity(self, i, j):
        sample1 = self.__samples[i]
        sample2 = self.__samples[j]

        sample1_token = sample1.token
        sample1_head_ent = sample1.head_entity
        sample1_head_ent_pos = sample1_head_ent["pos"]
        sample1_tail_ent = sample1.tail_entity
        sample1_tail_ent_pos = sample1_tail_ent["pos"]

        sample2_token = sample2.token
        sample2_head_ent = sample2.head_entity
        sample2_head_ent_token = sample2_token[
            sample2_head_ent["pos"][0] : sample2_head_ent["pos"][1]
        ]
        sample2_tail_ent = sample2.tail_entity
        sample2_tail_ent_token = sample2_token[
            sample2_tail_ent["pos"][0] : sample2_tail_ent["pos"][1]
        ]

        if sample1_head_ent_pos[1] - 1 < sample1_tail_ent_pos[0]:
            context1 = sample1_token[: sample1_head_ent_pos[0]]
            context2 = sample1_token[sample1_head_ent_pos[1] : sample1_tail_ent_pos[0]]
            context3 = sample1_token[sample1_tail_ent_pos[1] :]
            counterfactual_token = (
                context1
                + sample2_head_ent_token
                + context2
                + sample2_tail_ent_token
                + context3
            )
            start_index = len(context1)
            counterfactual_head_ent = {
                "pos": [start_index, start_index + len(sample2_head_ent_token)],
                "name": sample2_head_ent["name"],
            }
            start_index = len(context1) + len(sample2_head_ent_token) + len(context2)
            counterfactual_tail_ent = {
                "pos": [start_index, start_index + len(sample2_tail_ent_token)],
                "name": sample2_tail_ent["name"],
            }
            counterfactual_sample = CopyMRelSample(
                counterfactual_token,
                counterfactual_head_ent,
                counterfactual_tail_ent,
                sample1.image,
            )
            return counterfactual_sample
        elif sample1_head_ent_pos[0] > sample1_tail_ent_pos[1] - 1:
            context1 = sample1_token[: sample1_tail_ent_pos[0]]
            context2 = sample1_token[sample1_tail_ent_pos[1] : sample1_head_ent_pos[0]]
            context3 = sample1_token[sample1_head_ent_pos[1] :]
            counterfactual_token = (
                context1
                + sample2_tail_ent_token
                + context2
                + sample2_head_ent_token
                + context3
            )
            start_index = len(context1)
            counterfactual_tail_ent = {
                "pos": [start_index, start_index + len(sample2_tail_ent_token)],
                "name": sample2_tail_ent["name"],
            }
            start_index = len(context1) + len(sample2_tail_ent_token) + len(context2)
            counterfactual_head_ent = {
                "pos": [start_index, start_index + len(sample2_head_ent_token)],
                "name": sample2_head_ent["name"],
            }
            counterfactual_sample = CopyMRelSample(
                counterfactual_token,
                counterfactual_head_ent,
                counterfactual_tail_ent,
                sample1.image,
            )
            return counterfactual_sample
        else:
            return self.__exchange_image(i, j)

    def __exchange_image(self, i, j):
        sample1 = self.__samples[i]
        sample2 = self.__samples[j]

        sample1_token = sample1.token
        sample1_head_ent = sample1.head_entity
        sample1_tail_ent = sample1.tail_entity
        sample2_image = sample2.image

        counterfactual_sample = CopyMRelSample(
            sample1_token, sample1_head_ent, sample1_tail_ent, sample2_image
        )
        return counterfactual_sample

    def __convert_data(self, samples):
        parsed_data = {
            "indices": [],
            "segments": [],
            "pos_head": [],
            "pos_tail": [],
            "image": [],
        }
        for sample in samples:
            (
                indices,
                segments,
                pos_head,
                pos_tail,
            ) = self.__textProcessor.tokenize_sample(
                sample.token,
                copy.deepcopy(sample.head_entity["pos"]),
                copy.deepcopy(sample.tail_entity["pos"]),
            )
            parsed_data["indices"].append(indices)
            parsed_data["segments"].append(segments)
            parsed_data["pos_head"].append(pos_head)
            parsed_data["pos_tail"].append(pos_tail)
            parsed_data["image"].append(sample.image)
        return parsed_data

    def _generate_data(self):
        while True:
            real_set, counterfactual_set, label = self.__get_item()
            yield real_set["indices"], real_set["segments"], real_set[
                "pos_head"
            ], real_set["pos_tail"], real_set["image"], counterfactual_set[
                "indices"
            ], counterfactual_set[
                "segments"
            ], counterfactual_set[
                "pos_head"
            ], counterfactual_set[
                "pos_tail"
            ], counterfactual_set[
                "image"
            ], label

    def __get_item(self) -> Tuple:
        i, j = np.random.choice(self.__sample_idx, size=2, replace=False).tolist()
        sample = self.__samples[i]
        counterfactual_samples = []
        if self.__ex_img:
            counterfactual_sample = self.__exchange_image(i, j)
            counterfactual_samples.append(counterfactual_sample)
        if self.__ex_ent:
            counterfactual_sample = self.__exchange_entity(i, j)
            counterfactual_samples.append(counterfactual_sample)
        if self.__ex_cxt:
            counterfactual_sample = self.__exchange_context(i, j)
            counterfactual_samples.append(counterfactual_sample)

        real_set = self.__convert_data([sample])
        counterfactual_set = self.__convert_data(counterfactual_samples)
        label = [self.__dataset.relation2id(sample.relation)]
        return (real_set, counterfactual_set, label)


def get_train_dataloader(batch_size, dataset, textProcessor, ex_img, ex_ent, ex_cxt):
    generator = TrainDataGenerator(dataset, textProcessor, ex_img, ex_ent, ex_cxt)
    dataloader = Dataset.from_generator(
        generator._generate_data,
        tuple([tf.int64, tf.int64, tf.int64, tf.int64, tf.float32] * 2 + [tf.int64]),
        tuple(
            [
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None, None, None]),
            ]
            * 2
            + [tf.TensorShape([None])]
        ),
    )
    dataloader = dataloader.batch(batch_size)
    return iter(dataloader)


class ValidTestDataloader(object):
    def __init__(
        self, dataset: MRelDataset, textProcessor: TextProcessor, batch_size: int = 8
    ):
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__textProcessor = textProcessor
        self._valid_dataloader = self.__convert_data(dataset.Data("val"))
        self._test_dataloader = self.__convert_data(dataset.Data("test"))

    def Data(self, dtype: str):
        return getattr(self, f"_{dtype}_dataloader")

    def __convert_data(self, samples):
        parsed_data = {
            "indices": [],
            "segments": [],
            "pos_head": [],
            "pos_tail": [],
            "image": [],
            "label": [],
        }
        for sample in samples:
            (
                indices,
                segments,
                pos_head,
                pos_tail,
            ) = self.__textProcessor.tokenize_sample(
                sample.token, sample.head_entity["pos"], sample.tail_entity["pos"]
            )
            parsed_data["indices"].append(indices)
            parsed_data["segments"].append(segments)
            parsed_data["pos_head"].append(pos_head)
            parsed_data["pos_tail"].append(pos_tail)
            parsed_data["image"].append(sample.image)
            parsed_data["label"].append(self.__dataset.relation2id(sample.relation))
        dataloader = Dataset.from_tensor_slices(
            tuple([np.array(v) for v in parsed_data.values()])
        ).batch(self.__batch_size)
        return dataloader


if __name__ == "__main__":
    dataset = MRelDataset("../../FewShot-MNRE/dataset/MNRE")
    textProcessor = TextProcessor("../../FewShot-MNRE/pretrain/uncased_L-12_H-768_A-12")
    train_dataloader = get_train_dataloader(2, dataset, textProcessor)
    valid_test_dataloader = ValidTestDataloader(dataset, textProcessor, batch_size=2)

    for (
        real_ind,
        real_seg,
        real_ph,
        real_pt,
        real_img,
        cf_ind,
        cf_seg,
        cf_ph,
        cf_pt,
        cf_img,
        label,
    ) in train_dataloader:
        print(real_ind)
        print(real_seg)
        print(real_ph)
        print(real_pt)
        print(cf_ind)
        print(cf_seg)
        print(cf_ph)
        print(cf_pt)
        print(label)
        break

    for ind, seg, ph, pt, img, label in valid_test_dataloader.Data("valid"):
        print(ind)
        print(seg)
        print(ph)
        print(pt)
        print(img)
        print(label)
        break

    for ind, seg, ph, pt, img, label in valid_test_dataloader.Data("test"):
        print(ind)
        print(seg)
        print(ph)
        print(pt)
        print(img)
        print(label)
        break
