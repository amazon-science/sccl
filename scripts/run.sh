
python main.py \
    --result_path ./restest/searchsnippets/ \
    --num_classes 8 \
    --data_path ./datasamples/ \
    --dataname searchsnippets.csv \
    --dataset searchsnippets \
    --bert distil \
    --alpha 1 \
    --lr 1e-05 \
    --lr_scale 100 \
    --batch_size 400 \
    --temperature 0.5 \
    --base_temperature 0.07 \
    --max_iter 10 \
    --print_freq 250 \
    --seed 0 \
    --gpuid 0 &






