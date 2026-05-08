from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Ishara5/20bn-jester-event",
    repo_type="dataset",
    local_dir="./data",
    local_dir_use_symlinks=False
)
