#!/bin/bash
if [ $epoch_time ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=2000
fi

if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_acdc_out'
fi


if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='datasets/ACDC'
fi

if [ $num_classes ]; then
    NUM_CLASSES=$num_classes
else
    NUM_CLASSES=4
fi

if [ $learning_rate ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.05
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

if [ $batch_size ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=24
fi

if [ $multimask_output ]; then
    Multimask_Output=$batch_size
else
    Multimask_Output=True
fi

if [ $low_res ]; then
    Low_Res=$low_res
else
    Low_Res=56
fi

if [ $cfg ]; then
    CFG=$cfg
else
    CFG='configs/swin_tiny_patch4_window7_224_lite.yaml'
fi

echo "start train model"
python train.py --dataset ACDC --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --multimask_output $Multimask_Output --low_res $Low_Res \
--num_classes $NUM_CLASSES
