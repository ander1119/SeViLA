# parameters/data path
result_dir="option_with_definition_"

exp_name='timos_infer'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=3 python evaluate.py \
--cfg-path lavis/projects/sevila/eval/timos_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.timos.vis_processor.eval.n_frms=24 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'
