# Image dataloader
python3 tools/train_cube.py \
    --config configs/MoCA_cube_config.yaml \
    --train-loader image \
    --val-loader image \
    --train-datasets KITTI SUNRGBD Hypersim\
    --val-datasets KITTI SUNRGBD Hypersim\
    --train-epoch-length 50000 \
    --val-epoch-length 5000 \
    --no-compile

# WDS Dataloader
# python3 tools/train_cube.py \
#     --config configs/MoCA_cube_config.yaml \
#     --train-loader wds \
#     --val-loader wds \
#     --train-datasets KITTI SUNRGBD Hypersim\
#     --val-datasets KITTI SUNRGBD Hypersim\
#     --train-epoch-length 50000 \
#     --val-epoch-length 5000 \
#     --no-compile