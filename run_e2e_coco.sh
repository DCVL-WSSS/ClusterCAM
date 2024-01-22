GPU=0,1,2
session=exp
model=ClusterCAM_coco
coco_path=coco/COCO14

CUDA_VISIBLE_DEVICES=${GPU} python main.py --model deit_small_MCTformerV2_patch16_224 \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
                --input-size 448 \
                --num_workers 10 \
                --batch-size 6 \
                --epochs 60 \
                --lr 5e-4 \
                --data-set COCOSeg \
                --img-list coco \
                --data-path ${coco_path} \
                --label-file-path coco/cls_labels.npy \
                --layer-index 3 \
                --st_attn_depth 12 \
                --output_dir ${session}/${model} \
                --end_to_end \
                --scales 1.0 0.75 1.25 \
                --bg_thr 0.430 \
                --skip_seg 40