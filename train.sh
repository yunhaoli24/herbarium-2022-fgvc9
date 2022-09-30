#accelerate launch \
#--config_file train_config.yaml \
#main.py

python -m torch.distributed.launch --nproc_per_node 4 --use_env main.py