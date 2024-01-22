GPU=0,1,2
session=exp
model=ClusterCAM_voc
voc12_path=voc12/VOC2012

######## train MCTformer V2 ##########
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


attention_map_dir=attn-patchrefine
attention_map_npy_dir=attn-patchrefine-npy

# ######### Generating class-specific localization maps ##########
# CUDA_VISIBLE_DEVICES=0 python main.py --model deit_small_MCTformerV2_patch16_224 \
#                 --data-set VOC12MS \
#                 --scales 1.0 0.75 1.25 \
#                 --img-list ${voc12_path}/ImageSets/Segmentation \
#                 --data-path ${voc12_path} \
#                 --output_dir ${session}/${model} \
#                 --resume ${session}/${model}/checkpoint.pth \
#                 --gen_attention_maps \
#                 --attention-type fused \
#                 --st_attn_depth 12 \
#                 --layer-index 3 \
#                 --visualize-cls-attn \
#                 --attention-dir ${session}/${model}/${attention_map_dir} \
#                 --cam-npy-dir ${session}/${model}/${attention_map_npy_dir}


# ######### Evaluating the generated class-specific localization maps ##########
# CUDA_VISIBLE_DEVICES=${gen_GPU} python evaluation.py \
#                 --list ${voc12_path}/ImageSets/Segmentation/train_id.txt \
#                 --gt_dir ${voc12_path}/SegmentationClassAug \
#                 --logfile ${session}/${model}/${attention_map_npy_dir}/evallog.txt \
#                 --type npy \
#                 --curve True \
#                 --predict_dir ${session}/${model}/${attention_map_npy_dir} \
#                 --comment "train1464" \
#                 --start 30 \
#                 --end 99
