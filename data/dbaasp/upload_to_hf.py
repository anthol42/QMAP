from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os

load_dotenv("../.env")

#  Upload to hugginface
hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
if not hf_token:
    print("Huggin face token not found. Cannot push to hub. Exiting.")
    exit(1)
login(token=hf_token)

api = HfApi()
api.create_repo(
    repo_id="anthol42/qmap_benchmark_2025",
    repo_type="dataset",
    exist_ok=True
)

# Upload the 5 benchmark splits
for split in range(5):
    api.upload_file(
        path_or_fileobj=f'../build/benchmark_split_{split}.json',
        path_in_repo=f'benchmark_split_{split}.json',  # name in the repo
        repo_id="anthol42/qmap_benchmark_2025",
        repo_type="dataset"
    )

# Upload the full dataset
api.upload_file(
    path_or_fileobj='../build/dbaasp.json',
    path_in_repo='dbaasp.json',  # name in the repo
    repo_id="anthol42/qmap_benchmark_2025",
    repo_type="dataset"
)
