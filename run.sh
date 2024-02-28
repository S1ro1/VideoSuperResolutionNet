CWD="/home/xsirov00/BP_code"
DEBUG=${DEBUG:-0}
NAME=${NAME:-"test"}
CHECKPOINT="/home/xsirov00/BP_code/checkpoints/srresnet/epoch=89-step=135000.ckpt"


source $CWD/.venv/bin/activate
if [ "$DEBUG" -eq 1 ]; then
    EXTRA_ARGS="--limit-batches 0.01 --epochs 1 --log-every-n-steps 1 --check-val-every-n-epoch 1"
else
    EXTRA_ARGS="--epochs 1000 --log-every-n-steps 1 --check-val-every-n-epoch 5"
fi

if [ "$CHECKPOINT" != "" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --checkpoint-path $CHECKPOINT"
fi

python $CWD/main.py \
    --project BP --name $NAME --logger-type wandb \
    --train-hq $CWD/data/REDS/train/train_sharp --train-lq $CWD/data/REDS/train/train_sharp_bicubic/X4 \
    --val-hq $CWD/data/REDS/val/val_sharp --val-lq $CWD/data/REDS/val/val_sharp_bicubic/X4 \
    --batch-size 8 --num-workers 32 --devices 0  --save-top-k 3 $EXTRA_ARGS --devices 3
