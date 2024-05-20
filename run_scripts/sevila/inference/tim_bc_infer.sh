# parameters/data path
result_dir="outputs/"

exp_name='tim_zs_fullset_tmp'
# exp_name='test'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
# ckpt='/tmp2/adnchao/TropeVLM/SeViLA/lavis/outputs/tim_bc_ft_320_32_train_one_pos_one_neg_eval_20_trope_20_movie/checkpoint_best.pth'
# CUDA_VISIBLE_DEVICES=2 python evaluate.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:50 python -m torch.distributed.run --nproc_per_node=8 evaluate.py \
--cfg-path lavis/projects/sevila/eval/tim_bc_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=16 \
datasets.tim_bc.vis_processor.eval.n_frms=120 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='tim_bc'