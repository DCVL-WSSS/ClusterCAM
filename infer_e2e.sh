GPU=0
session=exp
model=ClusterCAM_voc
voc12_path=voc12/VOC2012

for idx in $(seq 79 -1 50)
do
    CUDA_VISIBLE_DEVICES=${GPU} python infer_e2e.py \
                        --weights ${session}/${model}/checkpoint_${idx}.pth \
                        --model deit_small_MCTformerV2_patch16_224 \
                        --list_path voc12/val_id.txt \
                        --gt_path ${voc12_path}/SegmentationClassAug\
                        --img_path ${voc12_path}/JPEGImages \
                        --save_path ${session}/${model}/seg_results_best/val_ms_crf_${idx} \
                        --save_path_c ${session}/${model}/seg_results_best/val_ms_crf_c_${idx} \
                        --scales 0.5 0.75 1.0 1.25 1.5 \
                        --use_crf True
    echo idx=${idx}
done

weight_path=ClusterCAM_VOC.pth

# CUDA_VISIBLE_DEVICES=${GPU} python infer_e2e.py \
#                     --weights ${weight_path} \
#                     --model deit_small_MCTformerV2_patch16_224 \
#                     --list_path voc12/val_id.txt \
#                     --gt_path ${voc12_path}/SegmentationClassAug\
#                     --img_path ${voc12_path}/JPEGImages \
#                     --save_path ${session}/${model}/seg_results_best/val_ms_crf_${idx} \
#                     --save_path_c ${session}/${model}/seg_results_best/val_ms_crf_c_${idx} \
#                     --scales 0.5 0.75 1.0 1.25 1.5 \
#                     --use_crf True