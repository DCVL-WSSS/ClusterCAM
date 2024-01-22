GPU=0
session=exp
model=ClusterCAM_coco
coco_path=coco/COCO14

for idx in $(seq 59 -1 40)
do
    CUDA_VISIBLE_DEVICES=${GPU} python infer_e2e_coco.py \
                        --weights ${session}/${model}/checkpoint_${idx}.pth \
                        --model deit_small_MCTformerV2_patch16_224 \
                        --list_path coco/val_id.txt \
                        --gt_path ${coco_path}/SegmentationClass/val2014\
                        --img_path ${coco_path}/val2014 \
                        --save_path ${session}/${model}/seg_results_best/val_ms_crf_${idx} \
                        --scales 0.5 0.75 1.0 1.25 1.5 \
                        --use_crf False #FIXME:
    echo idx=${idx}
done
