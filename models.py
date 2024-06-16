from huggingface_hub import snapshot_download

if __name__ == "__main__":
    snapshot_download(repo_id="BAAI/bge-reranker-v2-m3", local_dir="./reranker")
    snapshot_download(repo_id="BAAI/bge-m3", local_dir="./embedder")
    snapshot_download(repo_id="IlyaGusev/saiga_llama3_8b", revision="main_vllm", local_dir="./llm")