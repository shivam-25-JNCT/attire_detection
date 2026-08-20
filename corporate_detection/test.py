from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="miguelescamilla/id-card-yolo",
    filename="best.pt"
)

print("Model downloaded at:", model_path)