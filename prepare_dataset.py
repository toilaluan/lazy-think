import os
import json
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm

# Enable CUDA if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Create output directory
os.makedirs("data_output", exist_ok=True)

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

def compute_similarity_chunk(prompts_emb_chunk, thinks_emb, chunk_start, chunk_size):
    """
    Compute similarity matrix for a chunk of prompts
    """
    device = "cpu"
    
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

# Compute similarity matrix sequentially
logger.info("Computing similarity matrix...")
CHUNK_SIZE = 100  # Adjust based on GPU memory
similarity_matrix = np.zeros((len(PROMPTS_EMBEDDINGS), len(THINKS_EMBEDDINGS)))
for i in tqdm(range(0, len(PROMPTS_EMBEDDINGS), CHUNK_SIZE), desc="Computing similarity chunks"):
    end_idx = min(i + CHUNK_SIZE, len(PROMPTS_EMBEDDINGS))
    chunk = PROMPTS_EMBEDDINGS[i:end_idx]
    
    chunk_start, chunk_sims = compute_similarity_chunk(
        chunk,
        THINKS_EMBEDDINGS,
        i,
        CHUNK_SIZE
    )
    similarity_matrix[i:end_idx] = chunk_sims

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

# Process prompts sequentially in chunks
logger.info("Processing prompts...")
prepared_ds = []
for i in tqdm(range(0, len(PROMPTS), CHUNK_SIZE), desc="Processing chunks"):
    chunk_result = process_prompt_chunk(
        i,
        CHUNK_SIZE,
        similarity_matrix,
        PROMPTS,
        THINKS,
        SIM_TAGS
    )
    prepared_ds.extend(chunk_result)

# Save the prepared dataset
logger.info("Saving prepared dataset...")
with open(os.path.join("data_output", "prepared_ds.json"), "w") as f:
    json.dump(prepared_ds, f, indent=4)

logger.success("Dataset preparation completed!")

# Remove ray cleanup
torch.cuda.empty_cache()