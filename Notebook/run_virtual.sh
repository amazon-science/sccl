resdir="path-to-store-your-results"
datapath="path-to-your-data"

maxiter=100
bsize=400



python3 main.py \
    --resdir $resdir/ \
    --use_pretrain SBERT \
    --bert distilbert \
    --datapath $datapath \
    --dataname searchsnippets \
    --num_classes 8 \
    --text text \
    --label label \
    --objective SCCL \
    --augtype virtual \
    --temperature 0.5 \
    --eta 10 \
    --lr 1e-05 \
    --lr_scale 100 \
    --max_length 32 \
    --batch_size 400 \
    --max_iter $maxiter \
    --print_freq 100 \
    --gpuid 1 &

        




