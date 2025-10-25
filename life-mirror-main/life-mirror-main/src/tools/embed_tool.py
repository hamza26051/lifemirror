import os
import random
from .base import BaseTool, ToolInput, ToolResult

class EmbedTool(BaseTool):
    name = 'embed'

    def run(self, input: ToolInput) -> ToolResult:
        mode = os.getenv("LIFEMIRROR_MODE", "mock")
        dims = input.options.get("dims", 8)

        if mode == "mock":
            seed = hash(input.media_id) % (2**32)
            rng = random.Random(seed)
            vector = [rng.uniform(-1, 1) for _ in range(dims)]
            return ToolResult(
                success=True,
                data={"vector": vector, "model": "mock-embed-v1"}
            )

        # --- PROD MODE (Free alternative without paid APIs) ---
        try:
            import hashlib
            import numpy as np
            
            # Create a deterministic vector from URL using hash-based approach
            # This provides consistent embeddings without requiring paid APIs
            url_hash = hashlib.sha256(input.url.encode()).hexdigest()
            
            # Convert hash to vector of specified dimensions
            vector_size = input.options.get("dims", 384)  # Standard embedding size
            vector = []
            
            # Use hash chunks to create vector components
            for i in range(vector_size):
                # Take 8 characters from hash, cycling through
                chunk_start = (i * 8) % len(url_hash)
                chunk = url_hash[chunk_start:chunk_start + 8]
                # Convert hex to float between -1 and 1
                hex_val = int(chunk, 16) if len(chunk) == 8 else int(chunk.ljust(8, '0'), 16)
                normalized_val = (hex_val / (16**8)) * 2 - 1
                vector.append(normalized_val)
            
            # Normalize vector to unit length for cosine similarity
            vector = np.array(vector)
            vector = vector / np.linalg.norm(vector)
            
            return ToolResult(success=True, data={"vector": vector.tolist(), "model": "hash-based-free"})

        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))
