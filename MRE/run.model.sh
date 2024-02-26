seed=0
for n_train in {100..500..100}
do
    echo ---------- seed:$seed, n_train: $n_train ----------
    CUDA_VISIBLE_DEVICES=$1 python main.py --n_train $n_train --seed $seed --cf $2 --model $3
done