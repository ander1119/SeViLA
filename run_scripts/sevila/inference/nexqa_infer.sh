# parameters/data path
result_dir="testing"

exp_name='nextqa_infer'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node=1 evaluate.py \
--cfg-path lavis/projects/sevila/eval/nextqa_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.nextqa.vis_processor.eval.n_frms=32 \
run.batch_size_eval=8 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'
