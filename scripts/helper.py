import hashlib
import os

from modules import script_callbacks, ui_extra_networks, extra_networks, shared
import gradio as gr
from fastapi import FastAPI

root_path = os.getcwd()

folders = {
    "ti": os.path.join(root_path, "embeddings"),
    "hyper": os.path.join(root_path, "models", "hypernetworks"),
    "ckp": os.path.join(root_path, "models", "Stable-diffusion"),
    "lora": os.path.join(root_path, "models", "Lora"),
}
exts = (".bin", ".pt", ".safetensors", ".ckpt")
info_ext = ".info"
vae_suffix = ".vae"


def get_file_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# get cusomter model path
def get_custom_model_folder():
    global folders

    if shared.cmd_opts.embeddings_dir and os.path.isdir(shared.cmd_opts.embeddings_dir):
        folders["ti"] = shared.cmd_opts.embeddings_dir

    if shared.cmd_opts.hypernetwork_dir and os.path.isdir(shared.cmd_opts.hypernetwork_dir):
        folders["hyper"] = shared.cmd_opts.hypernetwork_dir

    if shared.cmd_opts.ckpt_dir and os.path.isdir(shared.cmd_opts.ckpt_dir):
        folders["ckp"] = shared.cmd_opts.ckpt_dir

    if shared.cmd_opts.lora_dir and os.path.isdir(shared.cmd_opts.lora_dir):
        folders["lora"] = shared.cmd_opts.lora_dir


def api_networks(_: gr.Blocks, app: FastAPI):
    @app.get("/diffusionhelper/hash")
    async def get_model_hash(modelType: str, name: str):
        baseFolder = folders[modelType]
        if not os.path.isdir(baseFolder):
            return {
                "error": "folder does not exist: " + baseFolder
            }
        for root, dirs, files in os.walk(baseFolder, followlinks=True):
            for file in files:
                if file.startswith(name):
                    return {
                        "hash": get_file_sha256(os.path.join(root, file))
                    }

    @app.get("/diffusionhelper/ping")
    async def ping():
        return {
            "pong": True
        }


script_callbacks.on_app_started(api_networks)
