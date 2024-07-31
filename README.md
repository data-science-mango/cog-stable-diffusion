# Stable Diffusion v2 Cog model

[![Replicate](https://replicate.com/stability-ai/stable-diffusion/badge)](https://replicate.com/stability-ai/stable-diffusion) 

This is an implementation of the [Diffusers Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict r8.im/data-science-mango/text-genai-sbxl-finte-tuned:latest \
        -i 'steps=50' \
        -i 'width=768' \
        -i 'height=768' \
        -i 'prompt="a photo of an astronaut riding a horse on mars"' \
        -i 'lora_scale=7.5' \
        -i 'num_images=1' \
        -i 'guidance_scale=7.5'

`Note`: lora weights were manually downloaded from s3 and copied locally before building the docker images