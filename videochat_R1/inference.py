from huggingface_hub import hf_hub_download
import torch, os
from tqdm import tqdm
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

cache_dir = './cache'
# Load the model in half-precision
model_path = "OpenGVLab/VideoChat-R1-thinking_7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="balanced", torch_dtype=torch.float16, cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)

# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")

def get_last_line(text: str) -> str:
    lines = text.split('\n')
    last_line = '\n'.join(lines[-1:])
    
    return last_line

def generate_answer(video_path, question):
    # video_path = "/fs/cml-projects/FMPT/Video-R1/downloads" + video_path[1:]
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "video", "path": video_path},
                ],
        },
    ]

    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, num_frames=10, tokenize=True, return_dict=True, return_tensors="pt").to('cuda')

    out = model.generate(**inputs, max_new_tokens=512)
    output = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    # return get_last_line(output)
    return output

SYS_PROMPT = '''{question}
            
            Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

            Then, provide your final answer within the <answer> </answer> tags.'''

csv_path = './z-outputs/videochat-r1-thinking.csv'
out_file = './z-outputs/videochat-r1-thinking.csv'

df = pd.read_csv(csv_path)
questions = list(df['Question'])
paths = list(df['video_path'])
results = ['' for i in range(len(questions))]
col_name = 'videochat-r1-thinking'

start_index = 10


print('Start index: ', start_index)

for i in tqdm(range(len(paths))):
    if i >= start_index:
        # answer = generate_answer(paths[i], questions[i], model, processor, accelerator)
        if type(questions[i]) == str and len(questions[i]) != 0:
            try:
                response = generate_answer(paths[i], SYS_PROMPT.replace('{question}', questions[i]))
                print('Response: ', response)
                results[i] = response
                df[col_name] = results
                df.to_csv(out_file)
            except Exception as e:
                    print('Question: ', questions[i])
                    print('Path: ', paths[i])
                    print('Error:', e)


