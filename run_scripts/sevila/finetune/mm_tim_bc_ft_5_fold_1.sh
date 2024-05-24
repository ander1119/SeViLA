# parameters
result_dir="outputs/"

exp_name='mm_tim_ft_fullset_5_fold_1'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --nproc_per_node=2 train.py \
--cfg-path lavis/projects/sevila/train/mm_tim_bc_ft_5_fold_1.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=16 \
datasets.mm_tim_bc.vis_processor.train.n_frms=120 \
datasets.mm_tim_bc.vis_processor.eval.n_frms=120 \
run.batch_size_train=1 \
run.batch_size_eval=1 \
run.init_lr=3e-5 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.accum_grad_iters=2 \
model.task='qvh_freeze_loc_train_qa_with_loc_train_qa_vid' \
model.finetuned=${ckpt} \
run.task='tim_bc'