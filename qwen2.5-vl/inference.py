from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
import torch, os
from tqdm import tqdm
from accelerate import Accelerator
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.video import VideoAsset
from transformers import AutoProcessor
global accelerator, model, processor
torch.cuda.empty_cache()
# Define the cache directory
# cache_dir = os.path.join(os.getcwd(), "model_cache")
SYS_PROMPT = '''You will be given a video and a question. Please provide a short phrase answer to the question based on the video.\nQuestion: {question}\nAnswer:'''
cache_dir = '/fs/clip-scratch/lizongxia'
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]
model_size = 32
MODELS = [f"Qwen/Qwen2.5-VL-{model_size}B-Instruct"]
csv_path = './round2_finalized.csv'
out_file = f'./z-outputs/qwen-VL-{model_size}B.csv'

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



for MODEL in MODELS:
    print('Processing Model ', MODEL)
    # Load the model with specified cache directory
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,  # Use half precision to save memory.
        device_map="balanced",       # Use a balanced device map for multi-GPU inference.
        low_cpu_mem_usage=True,      # Optimize CPU memory usage when loading the model.
        cache_dir=cache_dir
    ).eval()

    # Load the processor with specified cache directory
    processor = AutoProcessor.from_pretrained(
        MODEL,
        cache_dir=cache_dir
    )

    # accelerator = Accelerator(mixed_precision="fp16")

    # # Wrap the model using the accelerator.
    # # This ensures that your model and further inference calls will use all available GPUs.
    # model = accelerator.prepare(model)


    df = pd.read_csv(csv_path)
    questions = list(df['Question'])
    answers = list(df['Answer'])
    paths = list(df['video_path'])
    col_name = MODEL[5:]

    ### Test
    # print(generate_answer('./output.mp4', 'Descripbe the video'))
    

    if os.path.exists(out_file):
        af = pd.read_csv(out_file)
        results = list(af[col_name])
        for index in range(len(results)):
            if type(results[index]) != str or len(results[index])==0:
                start_index=index
                break
    else:
        results = ['' for ele in range(len(paths))]
        start_index = 0

    start_index=729
    print('Start index: ', start_index)

    for i in tqdm(range(len(paths))):
        if i >= start_index:
            # answer = generate_answer(paths[i], questions[i], model, processor, accelerator)
            print(f'Question: {questions[i]}')
            if type(questions[i]) == str and len(questions[i]) != 0:
                try:
                    answer = generate_answer(paths[i], questions[i])
                    results[i] = answer[0]

                    df[col_name] = results
                    df.to_csv(out_file)
                except Exception as e:
                    print('Question: ', questions[i])
                    print('Path: ', paths[i])
                    print('Error:', e)

    print(f'Finished Model {MODEL}')

