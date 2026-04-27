# Image training command for MoCA3D. Adjust the config file and dataset names as needed.
python3 tools/train.py --config configs/MoCA_config.yaml \
    --train-loader image --val-loader image \
    --train-datasets KITTI SUNRGBD Hypersim \
    --val-datasets KITTI SUNRGBD Hypersim \
    --train-epoch-length 50000 \
    --val-epoch-length 5000 \
    --no-compile

# Wds training command for MoCA3D. Adjust the config file and dataset names as needed.
# python3 tools/train.py --config configs/MoCA_config.yaml \
#     --train-loader wds --val-loader wds \
#     --train-datasets KITTI SUNRGBD Hypersim \
#     --val-datasets KITTI SUNRGBD Hypersim \
#     --train-epoch-length 50000 \
#     --val-epoch-length 5000 \
#     --no-compile