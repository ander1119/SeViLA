# parameters/data path
result_dir="outputs/"

exp_name='bc_with_definition_120_16'
# exp_name='test'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
--cfg-path lavis/projects/sevila/eval/tim_bc_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=16 \
datasets.tim_bc.vis_processor.eval.n_frms=120 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='tim_bc'