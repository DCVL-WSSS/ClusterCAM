GPU=0,1,2
session=exp
model=ClusterCAM_voc
voc12_path=voc12/VOC2012

CUDA_VISIBLE_DEVICES=${GPU} python main.py --model deit_small_MCTformerV2_patch16_224 \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
                --input-size 448 \
                --num_workers 10 \
                --batch-size 9 \
                --epochs 2 \
                --lr 5e-4 \
                --img-list voc12 \
                --data-path ${voc12_path} \
                --data-set VOC12Seg \
                --layer-index 3 \
                --st_attn_depth 12 \
                --output_dir ${session}/${model} \
                --end_to_end \
                --scales 1.0 0.75 1.25 \
                --bg_thr 0.4 \
                --skip_seg 1 \
                --ignore_delta 0.1
