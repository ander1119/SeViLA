# parameters
result_dir="outputs/"

exp_name='tim_bc_ft'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --nproc_per_node=2 train.py \
--cfg-path lavis/projects/sevila/train/tim_bc.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=16 \
datasets.tim_bc.vis_processor.train.n_frms=120 \
datasets.tim_bc.vis_processor.eval.n_frms=120 \
run.batch_size_train=1 \
run.batch_size_eval=1 \
run.init_lr=3e-5 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.accum_grad_iters=2 \
model.task='qvh_freeze_loc_train_qa_with_loc_train_qa_vid' \
model.finetuned=${ckpt} \
run.task='tim_bc'