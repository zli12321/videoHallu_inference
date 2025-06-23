# videoHallu_inferenve

Each folder contains a inference.py file that runs inference to generate responses for the corresponding model.

### Directions

Download video data and put them under ```new_videos_folder```.

```
pip install huggingface_hub

# Download data to your local dir
huggingface-cli download IntelligenceLab/VideoHallu --repo-type dataset --local-dir ./new_video_folders --local-dir-use-symlinks False
```

For example, if you want to run inference for Qwen-2.5-VL, do 
```
python ./qwen2.5-vl/inference.py
```

