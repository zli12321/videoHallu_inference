from huggingface_hub import hf_hub_download
import torch, os
from tqdm import tqdm
import pandas as pd
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

cache_dir = './cache'
# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir)
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", cache_dir=cache_dir)

# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")

def get_last_line(text: str) -> str:
    lines = text.split('\n')
    last_line = '\n'.join(lines[-1:])
    
    return last_line

def generate_answer(video_path, question):
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "video", "path": video_path},
                ],
        },
    ]

    inputs = processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to('cuda')

    out = model.generate(**inputs, max_new_tokens=60)
    output = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    return get_last_line(output)

SYS_PROMPT = '''You will be given a video and a question. Please provide a short phrase answer to the question based on the video.\nQuestion: {question}\nAnswer:'''
csv_path = './round2_finalized.csv'
out_file = './z-outputs/llavaNext.csv'

df = pd.read_csv(csv_path)
questions = list(df['Question'])
paths = list(df['video_path'])
results = ['' for i in range(len(questions))]
col_name = 'LLaVaNext'

start_index = 0


print('Start index: ', start_index)

for i in tqdm(range(len(paths))):
    if i >= start_index:
        # answer = generate_answer(paths[i], questions[i], model, processor, accelerator)
        if type(questions[i]) == str and len(questions[i]) != 0:
            try:
                response = generate_answer(paths[i], SYS_PROMPT.replace('{question}', questions[i]))
                results[i] = response
                df[col_name] = results
                df.to_csv(out_file)
            except Exception as e:
                    print('Question: ', questions[i])
                    print('Path: ', paths[i])
                    print('Error:', e)


