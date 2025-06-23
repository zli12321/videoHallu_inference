module load cuda

eval "$(micromamba shell hook --shell bash)"


cd /fs/nexus-scratch/zli12321/active-topic-modeling/video_understanding/qwen2.5-vl

CUDA_VISIBLE_DEVICES=0,1,2,3 python interactive_inference.py