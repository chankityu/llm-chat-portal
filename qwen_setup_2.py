from transformers import AutoProcessor, AutoModelForMultimodalLM
import torch, json
from qwen_vl_utils import process_vision_info

try:
    with open("chat_history.json") as f:
        messages = json.load(f)
except FileNotFoundError:
    print("Chat history not found, starting from scratch")


device = torch.device("cpu")
model = AutoModelForMultimodalLM.from_pretrained(
    #"Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="cpu"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def load_chat_history(path="chat_history.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant that can understand videos and images."}]
            }
        ]

def save_chat_history(messages, path="chat_history.json"):
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)


def add_user_message(text, video_path=None):
    content = []
    if video_path:
        content.append({"type": "video", "video": video_path})
    content.append({"type": "text", "text": text})

    messages.append({
        "role": "user",
        "content": content
    })

def run_chat(messages):
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

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def add_user_message(text, video_path=None):
    content = []
    if video_path:
        content.append({"type": "video", "video": video_path})
    content.append({"type": "text", "text": text})
    messages.append({"role": "user", "content": content})

def add_assistant_message(text):
    messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

messages = load_chat_history()

print("Welcome to interactive chat! Type 'exit' to quit.")

while True:
    user_input = input("\nYou Type 'exit' to quit.\n: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break

    # ask for video input
    video_path = input("Video path (or press Enter to skip): ").strip() or None

    add_user_message(user_input, video_path)

    reply = run_chat(messages)
    print("\nAssistant:", reply)

    add_assistant_message(reply)
    save_chat_history(messages)



