GPU=1
session=exp
model=ClusterCAM_voc

voc12_path=voc12/VOC2012
voc12_test_path=voc12/VOC12test/VOC2012

idx=66 # <- replace it with the best performing chekcpoint index

CUDA_VISIBLE_DEVICES=${GPU} python infer_e2e.py \
                    --weights ${session}/${model}/checkpoint_${idx}.pth \
                    --model deit_small_MCTformerV2_patch16_224 \
                    --list_path ${voc12_test_path}/ImageSets/Segmentation/test.txt \
                    --gt_path ${voc12_path}/SegmentationClassAug\
                    --img_path ${voc12_test_path}/JPEGImages \
                    --save_path ${session}/${model}/seg_results_best_test/val_ms_crf_${idx} \
                    --scales 0.5 0.75 1.0 1.25 1.5 \
                    --test \
                    --use_crf True
                    # --save_path_c ${session}/${model}/seg_results_best/val_ms_crf_c_${idx} \
echo idx=${idx}