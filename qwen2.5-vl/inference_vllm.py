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
SYS_PROMPT = '''You will be given a video and a question. Please provide a very BRIEF answer to the question based on the video.\nQuestion: {question}\nAnswer:'''
cache_dir = '/fs/clip-scratch/lizongxia'
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]
model_size = 7
MODELS = [f"Qwen/Qwen2.5-VL-{model_size}B-Instruct"]
csv_path = './round2_finalized.csv'
out_file = f'./A-outputs/round2/qwen-VL-{model_size}B.csv'

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
    print('Processing Model', MODEL)
    # 1) Engine configuration (instantiate once per MODEL)
    engine_args = EngineArgs(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
    )

    # 2) Prepare data
    df = pd.read_csv(csv_path)
    video_paths = df["video_path"].tolist()
    questions   = df["Question"].tolist()

    # 3) Vision preprocessing
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    # 4) Build all requests first
    requests = []
    for i, (vp, q) in enumerate(zip(video_paths, questions)):
        video_asset = VideoAsset(vp)  # vLLM multimodal schema :contentReference[oaicite:5]{index=5}
        prompt = (
            "<|im_start|>user\n"
            "<fim_prefix><|video|>" +
            q +
            "<fim_suffix>\n<|im_end|>\n<|im_start|>assistant\n"
        )
        requests.append({
            "prompt": prompt, 
            "multi_modal_data": {"video": video_asset},
            "sampling_params": sampling_params,
            "id": str(i),
        })

    # 5) Instantiate engine & generate _once_
    llm = LLM(
        MODEL,                          # 1st arg must be model name/path :contentReference[oaicite:1]{index=1}
        trust_remote_code=True,         # corresponds to EngineArgs.trust_remote_code :contentReference[oaicite:2]{index=2}
        dtype="bfloat16",               # EngineArgs.dtype :contentReference[oaicite:3]{index=3}
        tensor_parallel_size=2,         # EngineArgs.tensor_parallel_size :contentReference[oaicite:4]{index=4}
        max_model_len=4096,             # EngineArgs.max_model_len :contentReference[oaicite:5]{index=5}
        limit_mm_per_prompt={"video": 1}, # EngineArgs.limit_mm_per_prompt :contentReference[oaicite:6]{index=6}
    )
    # load model into GPU memory :contentReference[oaicite:6]{index=6}
    responses = llm.generate(requests) 

    # 6) Decode and store all outputs
    for r in responses:
        idx = int(r.request.id)
        gen_ids = r.outputs[0].ids
        text = llm.tokenizer.decode(gen_ids, 
                                    skip_special_tokens=True)
        df.at[idx, MODEL.replace("/", "_")] = text

    # 7) Save once per MODEL
    df.to_csv(out_file, index=False)
    print(f'Finished Model {MODEL}')