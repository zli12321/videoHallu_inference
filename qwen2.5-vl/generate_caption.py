import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os


cache_dir = '/fs/clip-scratch/lizongxia'

# Load the model with specified cache directory
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype="auto", 
    device_map="auto",
    cache_dir=cache_dir
)

# Load the processor with specified cache directory
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=cache_dir
)

# def generate_caption(video_path):
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "video",
#                     "video": video_path,
#                 },
#                 {"type": "text", "text": "Generate a detailed caption describing this video, including what entities are in the video, the attributes (colors, shape, texture, number, etc) of the entities and object, actions, event sequences, and what happened in the video."},
#             ],
#         }
#     ]

#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )

#     return output_text

def generate_caption(video_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": "Generate a detailed caption describing this video, including what entities are in the video, the attributes (colors, shape, texture, number, etc) of the entities and object, actions, event sequences, and what happened in the video."},
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

    # Inference: Generation of the output with increased max_new_tokens
    generated_ids = model.generate(**inputs, max_new_tokens=1024)  # Increased token limit
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def main():
    columns = ['Runawaygen2', 'Lavie', 'Veo2', 'pixverse', 'Kling', 'Sora', 'Cogvid']
    # columns = ['Lavie', 'Cogvid']
    df = pd.read_csv('./sorabench_data_with_local_paths.csv')
    prompts = list(df['Prompt'])


    for column in columns:
        paths = list(df[column])
        generated_captions = []
        for i in tqdm(range(len(paths))):
            path = paths[i]
            if type(paths[i]) == str and len(paths[i]) != 0:
                if column == 'Lavie':
                    path = '/fs/nexus-scratch/zli12321/active-topic-modeling/SoraBench/LaVie/res/SoraBen/' + path
                if column == 'Cogvid':
                    path = '/fs/nexus-scratch/zli12321/active-topic-modeling/SoraBench/CogVideo/out_videos/videos/' + path
                output_caption = generate_caption(path)
                generated_captions.append(output_caption)
                print('output: ', output_caption)
            else:
                generated_captions.append('')
            
        df[column+'_caption'] = generated_captions
        
        df.to_csv('./sorabench_captions.csv')

if __name__ == "__main__":
    main()