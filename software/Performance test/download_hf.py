from huggingface_hub import snapshot_download
# model_id = "MichaelWcc/llama_3.2_1B_IMU-based_HAR"
# snapshot_download(repo_id=model_id, local_dir="vicuna-hf",
#                   local_dir_use_symlinks=False, revision="main")
snapshot_download(repo_id="meta-llama/Llama-3.2-3B-Instruct", local_dir="llama3.2_3b", revision="main")
