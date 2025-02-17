import os
import json
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import ray
from loguru import logger
from tqdm import tqdm

# Enable CUDA if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Create output directory
os.makedirs("data_output", exist_ok=True)

# Initialize Ray with GPU support
ray.init(runtime_env={"pip": ["torch"]})

# 1. Load a pretrained Sentence Transformer model
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_MODEL.to(DEVICE)

# Similarity tag mapping
SIM_TAGS = {
    "0.1": "unrelated",
    "0.3": "somewhat-related", 
    "0.5": "related",
    "0.7": "very-related",
    "0.9": "extremely-related",
    "1.0": "identical",
}

def get_sim_tag(sim_score, sim_tags_dict):
    """
    Given a similarity score, returns the first tag whose threshold is not exceeded.
    """
    sim_tags_float = {float(k): v for k, v in sim_tags_dict.items()}
    for threshold in sorted(sim_tags_float.keys()):
        if sim_score <= threshold:
            return sim_tags_float[threshold]
    return "unknown"

def compute_embeddings_batch(texts, model, batch_size=32):
    """
    Compute embeddings in batches to optimize GPU memory usage
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch,
                convert_to_tensor=True,
                device=DEVICE
            )
            embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

@ray.remote(num_gpus=0.2)  # Adjust GPU fraction based on your needs
def compute_similarity_chunk(prompts_emb_chunk, thinks_emb, chunk_start, chunk_size):
    """
    Compute similarity matrix for a chunk of prompts using GPU
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert inputs to torch tensors
    thinks_emb_tensor = torch.tensor(thinks_emb).to(device)
    thinks_norms = torch.norm(thinks_emb_tensor, dim=1).to(device)
    
    # Process the chunk
    chunk_prompts = torch.tensor(prompts_emb_chunk).to(device)
    chunk_norms = torch.norm(chunk_prompts, dim=1).to(device)
    
    # Compute similarities for the chunk
    chunk_sims = torch.mm(chunk_prompts, thinks_emb_tensor.t())
    chunk_sims = chunk_sims / (chunk_norms.unsqueeze(1) * thinks_norms.unsqueeze(0))
    
    return chunk_start, chunk_sims.cpu().numpy()

# 2. Load and prepare the dataset
logger.info("Loading dataset...")
DS = load_dataset("KennethTang/DeepSeek-R1-only-CoT", split="train", num_proc=16)
DS = DS.select(range(10000))
logger.info(f"Loaded {len(DS)} samples")

logger.info("Extracting prompts and thinks...")
PROMPTS = [item["messages"][0]["content"] for item in DS]
THINKS = [item["messages"][1]["content"] for item in DS]

# Compute embeddings in batches
logger.info("Computing embeddings...")
THINKS_EMBEDDINGS = compute_embeddings_batch(THINKS, EMBEDDING_MODEL)
PROMPTS_EMBEDDINGS = compute_embeddings_batch(PROMPTS, EMBEDDING_MODEL)

# Distribute similarity matrix computation using Ray
logger.info("Computing distributed similarity matrix...")
CHUNK_SIZE = 100  # Adjust based on GPU memory
num_chunks = (len(PROMPTS_EMBEDDINGS) + CHUNK_SIZE - 1) // CHUNK_SIZE

# Launch distributed similarity computation tasks
similarity_tasks = []
for i in range(num_chunks):
    start_idx = i * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, len(PROMPTS_EMBEDDINGS))
    chunk = PROMPTS_EMBEDDINGS[start_idx:end_idx]
    
    task = compute_similarity_chunk.remote(
        chunk,
        THINKS_EMBEDDINGS,
        start_idx,
        CHUNK_SIZE
    )
    similarity_tasks.append(task)

# Collect and combine similarity matrix chunks
logger.info("Collecting similarity matrix chunks...")
similarity_matrix = np.zeros((len(PROMPTS_EMBEDDINGS), len(THINKS_EMBEDDINGS)))
for _ in tqdm(range(len(similarity_tasks)), desc="Processing similarity chunks"):
    chunk_start, chunk_sims = ray.get(similarity_tasks.pop(0))
    chunk_end = min(chunk_start + CHUNK_SIZE, len(PROMPTS_EMBEDDINGS))
    similarity_matrix[chunk_start:chunk_end] = chunk_sims

@ray.remote
def process_prompt_chunk(chunk_start, chunk_size, similarity_matrix, prompts, thinks, sim_tags):
    """
    Process a chunk of prompts using pre-computed similarity matrix
    """
    chunk_end = min(chunk_start + chunk_size, len(prompts))
    chunk_results = []
    
    for idx in range(chunk_start, chunk_end):
        sims = similarity_matrix[idx]
        data = {"prompt": prompts[idx], "thinks": {}}
        
        for tag in sim_tags.values():
            data["thinks"][tag] = []
        
        for sim_score, think in zip(sims, thinks):
            tag = get_sim_tag(sim_score, sim_tags)
            data["thinks"][tag].append(think)
            
        chunk_results.append(data)
    
    return chunk_results

# Process prompts in parallel using chunks
logger.info("Processing prompts in parallel...")
processing_tasks = [
    process_prompt_chunk.remote(
        i * CHUNK_SIZE,
        CHUNK_SIZE,
        similarity_matrix,
        PROMPTS,
        THINKS,
        SIM_TAGS
    )
    for i in range(num_chunks)
]

# Collect results
logger.info("Collecting results...")
prepared_ds = []
for chunk_result in tqdm(ray.get(processing_tasks), total=len(processing_tasks), desc="Processing chunks"):
    prepared_ds.extend(chunk_result)

# Save the prepared dataset
logger.info("Saving prepared dataset...")
with open(os.path.join("data_output", "prepared_ds.json"), "w") as f:
    json.dump(prepared_ds, f, indent=4)

logger.success("Dataset preparation completed!")

# Cleanup
ray.shutdown()
torch.cuda.empty_cache()