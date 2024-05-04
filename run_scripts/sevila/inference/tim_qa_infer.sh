# parameters/data path
result_dir="outputs/"

# exp_name='option_with_definition_256_16'
exp_name='test'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
--cfg-path lavis/projects/sevila/eval/tim_qa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=16 \
datasets.tim_qa.vis_processor.eval.n_frms=48 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'