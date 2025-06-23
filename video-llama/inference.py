import torch, os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
cache_dir = './cache'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    cache_dir=cache_dir
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def generate_answer(video_path, prompt):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 320}},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=256)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(response)

    return response




SYS_PROMPT = '''You will be given a video and a question. Please provide a short phrase answer to the question based on the video.\nQuestion: {question}\nAnswer:'''
csv_path = './round2_finalized.csv'
out_file = './z-outputs/videoLLama.csv'

df = pd.read_csv(csv_path)
questions = list(df['Question'])
paths = list(df['video_path'])
col_name = 'videoLLama'

if os.path.exists(out_file):
    df_existing = pd.read_csv(out_file)
    print(list(df_existing.keys()))
    print(col_name in list(df_existing.keys()))
    if col_name in list(df_existing.keys()):
        # Make sure to treat any NaN as an empty string.
        print('Existing column in such file')
        all_vals = list(df_existing[col_name])
        # print(all_vals)
        for y in range(len(all_vals)):
            if len(questions[i]) != 0:
                if type(all_vals[y])!=str or len(all_vals[y]) == 0:
                    start_index = y
                    break

        results = list(df_existing[col_name])
    else:
        results = ['' for ele in range(len(paths))]
        start_index = 0
else:
    results = ['' for ele in range(len(paths))]
    start_index = 0


print('Start index: ', start_index)

for i in tqdm(range(len(paths))):
    if i >= start_index:
        # answer = generate_answer(paths[i], questions[i], model, processor, accelerator)
        if type(questions[i]) == str and len(questions[i]) != 0:
            try:
                response = generate_answer(paths[i], SYS_PROMPT+questions[i])
                results[i] = response
                df[col_name] = results
                df.to_csv(out_file)
            except:
                print('Question: ', questions[i])
                print('Path: ', paths[i])