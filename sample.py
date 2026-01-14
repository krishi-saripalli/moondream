import argparse

import torch
from PIL import Image
from transformers import AutoModelForCausalLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument("--caption", action="store_true")
    args = parser.parse_args()

    image_path = args.image
    prompt = args.prompt

    model_id = "moondream/moondream3-preview"
    moondream = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    image = Image.open(image_path)

    settings = {"temperature": 0.5, "max_tokens": 768, "top_p": 0.3}

    if args.caption:
        print(moondream.caption(image, length="short", settings=settings))
    else:
        image_embeds = moondream.encode_image(image)
        answer = moondream.query(image, prompt, settings=settings, reasoning=True)
        print(answer)
