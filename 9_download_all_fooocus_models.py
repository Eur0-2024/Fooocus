# REQUIRES requests module!
# Badly written script to download all models fro Fooocus before using them inside ui. Made from version v2.5.0

import os
import requests
import subprocess


## all urls, model dirs and resulting filename was serached inside all files in Fooocus repo and prepared, sadly, by hand.
## it will NOT update with main repo updates

main_checkpoints = [
    
    {
        "url" : "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors",
        "model_dir" : "paths_checkpoints",
        "file_name" : "juggernautXL_v8Rundiffusion.safetensors"
    },
    
    {
        "url": "https://huggingface.co/mashb1t/fav_models/resolve/main/fav/playground-v2.5-1024px-aesthetic.fp16.safetensors",
        "model_dir": "paths_checkpoints",
        "file_name": "playground-v2.5-1024px-aesthetic.fp16.safetensors"
    },
    {
        "url": "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticStockPhoto_v20.safetensors",
        "model_dir": "paths_checkpoints",
        "file_name": "realisticStockPhoto_v20.safetensors"
    },
    {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors",
        "model_dir": "paths_checkpoints",
        "file_name": "sd_xl_base_1.0_0.9vae.safetensors"
    },
    {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors",
        "model_dir": "paths_checkpoints",
        "file_name": "sd_xl_refiner_1.0_0.9vae.safetensors"
    },
]

inpaint_checkpoints = [
    {
        "url": 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        "model_dir": "path_inpaint",
        "file_name": 'fooocus_inpaint_head.pth'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
        "model_dir": "path_inpaint",
        "file_name": 'inpaint.fooocus.patch'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
        "model_dir": "path_inpaint",
        "file_name": 'inpaint_v25.fooocus.patch'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
        "model_dir": "path_inpaint",
        "file_name": 'inpaint_v26.fooocus.patch'
    },

    {
        "url": 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
        "model_dir": "path_inpaint",
        "file_name": 'groundingdino_swint_ogc.pth'
    },

    {
        "url" : "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        "model_dir" : "path_inpaint",
        "file_name" : "isnet-general-use.onnx"
    },

    {
        "url" : "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
        "model_dir" : "path_inpaint",
        "file_name" : "silueta.onnx"
    },

    {
        "url" : "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
        "model_dir" : "path_inpaint",
        "file_name" : "u2net_human_seg.onnx"
    },

    {
        "url" : "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "model_dir" : "path_inpaint",
        "file_name" : "u2netp.onnx"
    },

    {
        "url" : "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "model_dir" : "path_inpaint",
        "file_name" : "u2net.onnx"
    },
]

lora_checkpoints = [
    {
        "url": "https://huggingface.co/mashb1t/fav_models/resolve/main/fav/SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors",
        "model_dir": "paths_loras",
        "file_name": "SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors"
    },

    {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors",
        "model_dir": "paths_loras",
        "file_name": "sd_xl_offset_example-lora_1.0.safetensors"
    },

    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        "model_dir": "paths_loras",
        "file_name": 'sdxl_lcm_lora.safetensors'
    },

    {
        "url": 'https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors',
        "model_dir": "paths_loras",
        "file_name": 'sdxl_lightning_4step_lora.safetensors'
    },

    {
        "url": 'https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors',
        "model_dir": "paths_loras",
        "file_name": 'sdxl_hyper_sd_4step_lora.safetensors'
    },
]

controlnet_checkpoints = [
    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        "model_dir": "path_controlnet",
        "file_name": 'control-lora-canny-rank128.safetensors'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        "model_dir": "path_controlnet",
        "file_name": 'fooocus_xl_cpds_128.safetensors'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        "model_dir": "path_controlnet",
        "file_name": 'fooocus_ip_negative.safetensors'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
        "model_dir": "path_controlnet",
        "file_name": 'ip-adapter-plus_sdxl_vit-h.bin'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
        "model_dir": "path_controlnet",
        "file_name": 'ip-adapter-plus-face_sdxl_vit-h.bin'
    },

    {
        "url" : "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "model_dir" : "path_controlnet",
        "file_name" : "detection_Resnet50_Final.pth"
    },

    {
        "url" : "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "model_dir" : "path_controlnet",
        "file_name" : "parsing_parsenet.pth"
    },
]

upscale_checkpoints = [
    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        "model_dir": "path_upscale_models",
        "file_name": 'fooocus_upscaler_s409985e5.bin'
    },
]

safety_checker_chekpoints = [
    {
        "url": 'https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin',
        "model_dir": "path_safety_checker",
        "file_name": 'stable-diffusion-safety-checker.bin'
    },
]

sam_checkpoints = [
    {
        "url": 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth',
        "model_dir": "path_sam",
        "file_name": 'sam_vit_b_01ec64.pth'
    },

    {
        "url": 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth',
        "model_dir": "path_sam",
        "file_name": 'sam_vit_l_0b3195.pth'
    },

    {
        "url": 'https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth',
        "model_dir": "path_sam",
        "file_name": 'sam_vit_h_4b8939.pth'
    },
]

vae_approx_checkpoints = [
    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth',
        "model_dir": "path_vae_approx",
        "file_name": 'xlvaeapp.pth'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt',
        "model_dir": "path_vae_approx",
        "file_name": 'vaeapp_sd15.pth'
    },

    {
        "url": 'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors',
        "model_dir": "path_vae_approx",
        "file_name": 'xl-to-v1_interposer-v4.0.safetensors'
    },
]

expansion_chekpoints = [
    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        "model_dir": "path_fooocus_expansion",
        "file_name": 'pytorch_model.bin'
    },
]

clip_vision_checkpoints = [
    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        "model_dir": "path_clip_vision",
        "file_name": 'clip_vision_vit_h.safetensors'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/wd-v1-4-moat-tagger-v2.onnx',
        "model_dir": "path_clip_vision",
        "file_name": 'wd-v1-4-moat-tagger-v2.onnx'
    },

    {
        "url": 'https://huggingface.co/lllyasviel/misc/resolve/main/wd-v1-4-moat-tagger-v2.csv',
        "model_dir": "path_clip_vision",
        "file_name": 'wd-v1-4-moat-tagger-v2.csv'
    },
]

extra_checkpoints = [
    {
        "url": 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        "model_dir": "path_extra_facex",
        "file_name": 'detection_Resnet50_Final.pth'
    },

    {
        "url": 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth',
        "model_dir": "path_extra_facex",
        "file_name": 'detection_mobilenet0.25_Final.pth'
    },

    {
        "url": 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth',
        "model_dir": "path_extra_facex",
        "file_name": 'parsing_bisenet.pth'
    },

    {
        "url": 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
        "model_dir": "path_extra_facex",
        "file_name": 'parsing_parsenet.pth'
    },
]

paths = {
    "paths_checkpoints": './Fooocus/models/checkpoints/',
    "paths_loras": './Fooocus/models/loras/',
    "path_embeddings": './Fooocus/models/embeddings/',
    "path_vae_approx": './Fooocus/models/vae_approx/',
    "path_vae": './Fooocus/models/vae/',
    "path_upscale_models": './Fooocus/models/upscale_models/',
    "path_inpaint": './Fooocus/models/inpaint/',
    "path_controlnet": './Fooocus/models/controlnet/',
    "path_clip_vision": './Fooocus/models/clip_vision/',
    "path_fooocus_expansion": './Fooocus/models/prompt_expansion/fooocus_expansion/',
    "path_wildcards": './Fooocus/wildcards/',
    "path_safety_checker": './Fooocus/models/safety_checker/',
    "path_sam": './Fooocus/models/sam/',
    "path_extra_facex": './Fooocus/extras/facexlib/weights/',
}


def resolve_url(url):
    response = requests.head(url, allow_redirects=True)
    return response.url

def download_models(data):
    for item in data:
        # get dicts one by one
        url_base = item["url"]
        model_dir = item["model_dir"]
        model_dir_resolve = paths.get(model_dir, "Unknown-path")
        file_name = item["file_name"]

        out_path = os.path.join(model_dir_resolve, file_name)

        if not os.path.exists(model_dir_resolve):
            os.makedirs(model_dir_resolve)

        if not os.path.exists(out_path):
            url_fin = resolve_url(url_base)
            print(f"Downloading {file_name} to {out_path}...")
            subprocess.run(['wget', '-O', out_path, url_fin])
        else:
            print(f"{file_name} already exists at {out_path}, skipping download.")

        # dry_run = f'{file_name} -> target:{out_path}\n{url_fin}\nmodel_dir : {model_dir}\nmodel_dir_resolve:{model_dir_resolve}\n'
        # print(dry_run)


# run
if __name__ == "__main__":
    print("trying to download models...")
    download_models(main_checkpoints)
    download_models(inpaint_checkpoints)
    download_models(lora_checkpoints)
    download_models(controlnet_checkpoints)
    download_models(upscale_checkpoints)
    download_models(safety_checker_chekpoints)
    download_models(sam_checkpoints)
    download_models(vae_approx_checkpoints)
    download_models(expansion_chekpoints)
    download_models(clip_vision_checkpoints)
    download_models(extra_checkpoints)
