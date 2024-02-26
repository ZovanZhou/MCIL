import os
import random
import argparse
from data_loader import *
from model import *
from utils import train_model, evaluate_model
import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--n_train", type=int, default=100)
parser.add_argument("--cf", type=int, choices=[0, 1], default=0)
parser.add_argument("--ex_img", type=int, choices=[0, 1], default=1)
parser.add_argument("--ex_ent", type=int, choices=[0, 1], default=1)
parser.add_argument("--ex_cxt", type=int, choices=[0, 1], default=1)
parser.add_argument("--model", type=str, choices=["umt", "mkgformer"], default="umt")
args = parser.parse_args()

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

n_cf = 0
if args.ex_img:
    n_cf += 1
if args.ex_ent:
    n_cf += 1
if args.ex_cxt:
    n_cf += 1

train_iter = args.n_train // args.batch_size

bert_path = "../pretrain/uncased_L-12_H-768_A-12"
dataset = MRelDataset("../dataset/MRE", n_train=args.n_train)
RELATION = dataset.RELATION

textProcessor = TextProcessor(bert_path=bert_path)
train_dataloader = get_train_dataloader(
    args.batch_size,
    dataset,
    textProcessor,
    args.ex_img,
    args.ex_ent,
    args.ex_cxt,
)
valid_test_dataloader = ValidTestDataloader(
    dataset, textProcessor, batch_size=args.batch_size
)

sentence_encoder = BERTEmbedding(bert_path=bert_path, fine_tune=True)
if args.model == "umt":
    model = UMT(
        sentence_encoder=sentence_encoder, n_label=len(RELATION), use_cf=args.cf
    )
elif args.model == "mkgformer":
    model = MKGformer(
        sentence_encoder=sentence_encoder, n_label=len(RELATION), use_cf=args.cf
    )
optimizer = tf.optimizers.Adam(learning_rate=args.lr)

max_f1 = 0.0
max_result = {}

for _ in range(args.epoch):
    train_model(train_dataloader, model, optimizer, train_iter, cf=args.cf, n_cf=n_cf)
    val_result = evaluate_model(valid_test_dataloader.Data("valid"), model, dataset)
    if val_result["micro_f1"] >= max_f1:
        max_f1 = val_result["micro_f1"]
        test_result = evaluate_model(valid_test_dataloader.Data("test"), model, dataset)
        max_result = {"valid": val_result, "test": test_result}
        pprint.pprint(max_result)
        model.save_weights(
            f"./weights/mre-{args.model}-cf-{args.cf}-n-{args.n_train}.h5"
        )

print("\n\n")
pprint.pprint(max_result)

with open(
    f"./results/{args.model}-cf-{args.cf}-train-{args.n_train}-seed-{args.seed}.txt",
    "w",
) as fw:
    pprint.pprint(max_result, stream=fw)
