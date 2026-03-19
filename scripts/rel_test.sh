#!/bin/bash

export OMP_NUM_THREADS=1
export gpu_num=1
export CUDA_VISIBLE_DEVICES="0"


archive_dir="/root/autodl-tmp/ckpt"
# 需要删除这个目录下的 last_checkpoint，不然不会使用你指定的权重
python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num  \
  tools/relation_test_net.py \
  --config-file "$archive_dir/config.yml" \
  TEST.IMS_PER_BATCH $[$gpu_num] \
  MODEL.WEIGHT  "$archive_dir/model_0014000.pth" \
  MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
  DATASETS.VG_TEST "('VG_stanford_filtered_with_attribute_test', )" \
  TEST.ALLOW_LOAD_FROM_CACHE True