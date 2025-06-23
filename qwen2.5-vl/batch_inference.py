import os
import torch
import pandas as pd
from multiprocessing import Pool
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

SYS_PROMPT = '''You will be given a video and a question. Please provide a very BRIEF answer to the question based on the video.\nQuestion: {question}\nAnswer:'''
cache_dir = '/fs/clip-scratch/lizongxia'
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"  # Pick one for example.
cache_dir = '/fs/clip-scratch/lizongxia'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, 
        torch_dtype=torch.float16, 
        device_map="balanced", 
        # torch_dtype=torch.float16,
        cache_dir=cache_dir
    )

    # Load the processor with specified cache directory
processor = AutoProcessor.from_pretrained(
        MODEL,
        cache_dir=cache_dir
)




def generate_answer(video_path, question):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {"type": "text", "text":SYS_PROMPT.replace('{question}', question)},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)

        return output_text

# Function to run inference on a subset of the data assigned to one GPU.
def process_inference(args):
    gpu_id, model_name, csv_subset, cache_dir, col_name = args

    # Set the specific GPU for this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load model and processor on this GPU.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    results = ['' for _ in range(len(csv_subset))]
    for i, row in enumerate(csv_subset.itertuples()):
        video_path = row.video_path
        question = row.Question
        answer = generate_answer(video_path, question)  # your existing function
        results[i] = answer[0]
    csv_subset[col_name] = results
    csv_subset.to_csv(f"./A-outputs/round1_gpu{gpu_id}.csv", index=False)
    return f"Process on GPU {gpu_id} finished."

if __name__ == '__main__':
    csv_path = './round1.csv'
    df = pd.read_csv(csv_path)
    num_gpus = 4
    # Example: partition dataframe into num_gpus parts.
    df_parts = [df.iloc[i::num_gpus].copy() for i in range(num_gpus)]
    
    col_name = MODEL[5:]

    args_list = []
    for gpu_id in range(num_gpus):
        args = (gpu_id, MODEL, df_parts[gpu_id], cache_dir, col_name)
        args_list.append(args)

    with Pool(num_gpus) as p:
        results = p.map(process_inference, args_list)
    
    print(results)


