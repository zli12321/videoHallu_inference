from transformers import AutoProcessor, AutoModelForImageTextToText
import torch, os
from tqdm import tqdm
import pandas as pd

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
cache_dir = './cache'
SYS_PROMPT = '''You will be given a video and a question. Please provide a short phrase answer to the question based on the video.\nQuestion: {question}\nAnswer:'''
csv_path = './round2_finalized.csv'
out_file = './z-outputs/smolLVML.csv'


processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
    cache_dir=cache_dir
).to("cuda")

def generate_answer(video_path, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": question}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # print(generated_texts[0])
    return generated_texts[0]

df = pd.read_csv(csv_path)
questions = list(df['Question'])
paths = list(df['video_path'])
col_name = 'SmoilLVLM'

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
            if len(questions[y]) != 0:
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

