# MoCA evaluation script
# python3 tools/evaluate.py \
#    --config configs/MoCA_config.yaml \
#    --loader wds \
#    --datasets KITTI SUNRGBD Hypersim

python3 tools/evaluate.py \
   --config configs/MoCA_config.yaml \
   --loader image \
   --datasets KITTI SUNRGBD Hypersim
