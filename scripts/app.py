import sys

sys.path.append(".")
import argparse
import gradio as gr
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
from scedit_pytorch import UNet2DConditionModel, load_scedit_into_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="pretrained model path",
    )
    parser.add_argument(
        "--scedit_name_or_path", type=str, required=True, help="ziplora path"
    )
    parser.add_argument("--scale", type=float, default=1.0, help="weight scale")
    return parser.parse_args()


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
# load unet with sctuner
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet"
)
unet.set_sctuner(scale=args.scale)
unet = load_scedit_into_unet(args.scedit_name_or_path, unet)
# load pipeline
pipeline = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, unet=unet
)
pipeline = pipeline.to(device=device, dtype=torch.float16)


def run(prompt: str):
    # generator = torch.Generator(device="cuda").manual_seed(42)
    generator = None
    image = pipeline(prompt=prompt, generator=generator).images[0]
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Text(label="prompt", value="A picture of a sbu dog in a bucket")
            bttn = gr.Button(value="Run")
        with gr.Column():
            out = gr.Image(label="out")
    prompt.submit(fn=run, inputs=[prompt], outputs=[out])
    bttn.click(fn=run, inputs=[prompt], outputs=[out])

    demo.launch(share=True, debug=True, show_error=True)
