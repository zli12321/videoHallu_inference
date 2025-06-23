import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Define the cache directory
cache_dir = '/fs/clip-scratch/lizongxia'

# Load the model with a device map that balances across GPUs.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.float16,  # Use half precision to save memory.
    device_map="balanced",       # Use a balanced device map for multi-GPU inference.
    low_cpu_mem_usage=True,      # Optimize CPU memory usage when loading the model.
    cache_dir=cache_dir
)

# Load the processor with the specified cache directory
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=cache_dir
)

# Do not call model.to(device) when using device_map.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Interactive Qwen2.5-VL session. Type 'exit' to quit.\n")

while True:
    user_question = input("Enter your question: ").strip()
    if user_question.lower() == "exit":
        break

    # Build the chat message structure (text and video)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    # "video": "/fs/nexus-scratch/zli12321/active-topic-modeling/video_understanding/picks/runaway_feather.mp4",
                    "video": "/fs/nexus-scratch/zli12321/active-topic-modeling/video_understanding/picks/watermelon_explode.mp4",
                },
                {"type": "text", "text": user_question},
            ],
        }
    ]
    
    # Apply the chat template to format the prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    print("-" * 50)