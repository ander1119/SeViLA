# parameters/data path
result_dir="outputs/"

exp_name='tim_bc_zeroshot'
# exp_name='test'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
# ckpt='/mnt/ssd2/TropeVLM/SeViLA/lavis/outputs/tim_bc_ft_train_one_pos_one_neg_eval_20_trope_20_movie/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 evaluate.py \
--cfg-path lavis/projects/sevila/eval/tim_bc_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=16 \
datasets.tim_bc.vis_processor.eval.n_frms=120 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='tim_bc'