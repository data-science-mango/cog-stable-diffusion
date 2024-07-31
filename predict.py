import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from compel import Compel, ReturnedEmbeddingsType

# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "SG161222/RealVisXL_V4.0"
MODEL_CACHE = "diffusers-cache"
LOCAL_DIR = "pytorch_lora_weights.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, cache_dir=MODEL_CACHE, torch_dtype=torch.float16, use_safetensors=True
        ).to(self.device)
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)

        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="Double nose",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        num_images: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        lora_scale: float = Input(
            description="Scale for Lora", ge=1, le=20, default=7.5
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=42
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        ### Inferencia------------------------------
        conditioning, pooled = self.compel(prompt)
        negative_conditioning, negative_pooled = self.compel(negative_prompt)
        [conditioning, negative_conditioning] = (
            self.compel.pad_conditioning_tensors_to_same_length(
                [conditioning, negative_conditioning]
            )
        )

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt_embeds=conditioning,
            negative_prompt_embeds=negative_conditioning,
            pooled_prompt_embeds=pooled,
            negative_pooled_prompt_embeds=negative_pooled,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            cross_attention_kwargs={"scale": lora_scale},
            generator=generator,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
