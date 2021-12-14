resdir="path-to-store-your-results"
datapath="path-to-your-data"

bsize=400
maxiter=1000


python3 main.py \
    --resdir $resdir/ \
    --use_pretrain SBERT \
    --bert distilbert \
    --datapath $datapath \
    --dataname searchsnippets_trans_subst_20 \
    --num_classes 8 \
    --text text \
    --label label \
    --objective SCCL \
    --augtype explicit \
    --temperature 0.5 \
    --eta 10 \
    --lr 1e-05 \
    --lr_scale 100 \
    --max_length 32 \
    --batch_size $bsize \
    --max_iter $maxiter \
    --print_freq 100 \
    --gpuid 7 &
            



