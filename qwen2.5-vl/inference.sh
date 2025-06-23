module load cuda

eval "$(micromamba shell hook --shell bash)"

cd /fs/nexus-scratch/zli12321/active-topic-modeling/video_understanding

micromamba activate videoscore

export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./qwen2.5-vl/inference.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./qwen2.5-vl/interactive_inference.py