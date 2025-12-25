# from diffusers import StableDiffusionPipeline
# import torch
#
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",  # 模型名称
#     torch_dtype=torch.float16
# ).to("cuda")
#
# prompt = "A fantasy castle on a mountain, sunset"
# image = pipe(prompt).images[0]
# image.save("castle.png")

# %%
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="cuda"
)
pipe.enable_attention_slicing()

image = pipe(
    "Astronaut in a jungle, cold color palette, muted colors, detailed",
    num_inference_steps=25
).images[0]

# %%
