import numpy as np
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentInput, AgentOutput

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class EmbedderAgent(BaseAgent):
    """Agent for creating and managing vector embeddings"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.index = None
        self.stored_embeddings = []
        self.stored_metadata = []
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use a lightweight, fast model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Initialized SentenceTransformer with dimension {self.embedding_dim}")
            else:
                self.logger.warning("SentenceTransformers not available, using fallback embedding")
                self.model = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def run(self, input: AgentInput) -> AgentOutput:
        """Create embeddings for text or perform similarity search"""
        try:
            operation = input.context.get('operation', 'embed')
            
            if operation == 'embed':
                return self._create_embeddings(input)
            elif operation == 'search':
                return self._similarity_search(input)
            elif operation == 'compare':
                return self._compare_embeddings(input)
            elif operation == 'cluster':
                return self._cluster_embeddings(input)
            else:
                return self._create_output(
                    success=False,
                    data={},
                    error=f"Unknown operation: {operation}",
                    confidence=0.0
                )
                
        except Exception as e:
            return self._handle_error(e, input)
    
    def _create_embeddings(self, input: AgentInput) -> AgentOutput:
        """Create embeddings for given text(s)"""
        try:
            texts = input.context.get('texts', [])
            text = input.context.get('text', '')
            
            # Handle single text or list of texts
            if text and not texts:
                texts = [text]
            elif not texts:
                return self._create_output(
                    success=False,
                    data={},
                    error="No text provided for embedding",
                    confidence=0.0
                )
            
            # Create embeddings
            embeddings_result = self._generate_embeddings(texts)
            
            # Store embeddings if requested
            store_embeddings = input.context.get('store_embeddings', False)
            if store_embeddings and embeddings_result['success']:
                self._store_embeddings(
                    embeddings_result['embeddings'],
                    texts,
                    input.context.get('metadata', {})
                )
            
            return self._create_output(
                success=embeddings_result['success'],
                data=embeddings_result,
                confidence=embeddings_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _similarity_search(self, input: AgentInput) -> AgentOutput:
        """Perform similarity search against stored embeddings"""
        try:
            query_text = input.context.get('query_text', '')
            top_k = input.context.get('top_k', 5)
            
            if not query_text:
                return self._create_output(
                    success=False,
                    data={},
                    error="No query text provided",
                    confidence=0.0
                )
            
            if not self.stored_embeddings:
                return self._create_output(
                    success=False,
                    data={},
                    error="No stored embeddings available for search",
                    confidence=0.0
                )
            
            # Generate query embedding
            query_embedding_result = self._generate_embeddings([query_text])
            
            if not query_embedding_result['success']:
                return self._create_output(
                    success=False,
                    data=query_embedding_result,
                    error="Failed to generate query embedding",
                    confidence=0.0
                )
            
            query_embedding = query_embedding_result['embeddings'][0]
            
            # Perform similarity search
            search_results = self._perform_similarity_search(
                query_embedding, top_k
            )
            
            return self._create_output(
                success=True,
                data={
                    'query_text': query_text,
                    'results': search_results,
                    'total_stored': len(self.stored_embeddings)
                },
                confidence=0.8
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _compare_embeddings(self, input: AgentInput) -> AgentOutput:
        """Compare two sets of embeddings"""
        try:
            texts1 = input.context.get('texts1', [])
            texts2 = input.context.get('texts2', [])
            
            if not texts1 or not texts2:
                return self._create_output(
                    success=False,
                    data={},
                    error="Need two sets of texts for comparison",
                    confidence=0.0
                )
            
            # Generate embeddings for both sets
            embeddings1_result = self._generate_embeddings(texts1)
            embeddings2_result = self._generate_embeddings(texts2)
            
            if not (embeddings1_result['success'] and embeddings2_result['success']):
                return self._create_output(
                    success=False,
                    data={},
                    error="Failed to generate embeddings for comparison",
                    confidence=0.0
                )
            
            # Calculate similarities
            similarities = self._calculate_pairwise_similarities(
                embeddings1_result['embeddings'],
                embeddings2_result['embeddings']
            )
            
            # Generate comparison analysis
            comparison_analysis = self._analyze_similarities(
                similarities, texts1, texts2
            )
            
            return self._create_output(
                success=True,
                data={
                    'similarities': similarities,
                    'analysis': comparison_analysis,
                    'texts1': texts1,
                    'texts2': texts2
                },
                confidence=0.8
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _cluster_embeddings(self, input: AgentInput) -> AgentOutput:
        """Cluster stored embeddings"""
        try:
            n_clusters = input.context.get('n_clusters', 3)
            
            if not self.stored_embeddings:
                return self._create_output(
                    success=False,
                    data={},
                    error="No stored embeddings available for clustering",
                    confidence=0.0
                )
            
            # Perform clustering
            clustering_result = self._perform_clustering(n_clusters)
            
            return self._create_output(
                success=clustering_result['success'],
                data=clustering_result,
                confidence=clustering_result.get('confidence', 0.7)
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings for a list of texts"""
        try:
            if self.model and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use SentenceTransformer
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                
                return {
                    'success': True,
                    'embeddings': embeddings.tolist(),
                    'method': 'sentence_transformer',
                    'model_name': self.model._modules['0'].auto_model.name_or_path,
                    'embedding_dim': self.embedding_dim,
                    'confidence': 0.9
                }
            else:
                # Fallback to simple hash-based embeddings
                embeddings = self._create_fallback_embeddings(texts)
                
                return {
                    'success': True,
                    'embeddings': embeddings,
                    'method': 'fallback_hash',
                    'model_name': 'hash_based',
                    'embedding_dim': self.embedding_dim,
                    'confidence': 0.3
                }
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'embeddings': [],
                'method': 'failed',
                'confidence': 0.0
            }
    
    def _create_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create simple hash-based embeddings as fallback"""
        embeddings = []
        
        for text in texts:
            # Create a simple embedding based on character frequencies
            embedding = [0.0] * self.embedding_dim
            
            # Use character frequencies and positions
            for i, char in enumerate(text.lower()):
                if char.isalnum():
                    char_code = ord(char)
                    # Distribute character influence across embedding dimensions
                    for j in range(min(10, self.embedding_dim)):
                        idx = (char_code + i + j) % self.embedding_dim
                        embedding[idx] += 1.0 / (len(text) + 1)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings
    
    def _store_embeddings(self, embeddings: List[List[float]], texts: List[str], metadata: Dict):
        """Store embeddings with metadata"""
        try:
            for i, (embedding, text) in enumerate(zip(embeddings, texts)):
                self.stored_embeddings.append(embedding)
                self.stored_metadata.append({
                    'text': text,
                    'index': len(self.stored_embeddings) - 1,
                    'timestamp': metadata.get('timestamp'),
                    'source': metadata.get('source', 'unknown'),
                    'additional_metadata': metadata.get('additional_metadata', {})
                })
            
            # Update FAISS index if available
            if FAISS_AVAILABLE:
                self._update_faiss_index()
                
            self.logger.info(f"Stored {len(embeddings)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to store embeddings: {e}")
    
    def _update_faiss_index(self):
        """Update FAISS index with stored embeddings"""
        try:
            if not self.stored_embeddings:
                return
            
            # Create or update FAISS index
            embeddings_array = np.array(self.stored_embeddings, dtype=np.float32)
            
            if self.index is None:
                # Create new index
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            else:
                # Reset index
                self.index.reset()
            
            # Add embeddings to index
            self.index.add(embeddings_array)
            
            self.logger.info(f"Updated FAISS index with {len(self.stored_embeddings)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to update FAISS index: {e}")
    
    def _perform_similarity_search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        try:
            if FAISS_AVAILABLE and self.index is not None:
                # Use FAISS for efficient search
                query_array = np.array([query_embedding], dtype=np.float32)
                scores, indices = self.index.search(query_array, min(top_k, len(self.stored_embeddings)))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.stored_metadata):
                        result = self.stored_metadata[idx].copy()
                        result['similarity_score'] = float(score)
                        results.append(result)
                
                return results
            else:
                # Fallback to manual similarity calculation
                similarities = []
                
                for i, stored_embedding in enumerate(self.stored_embeddings):
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    similarities.append((similarity, i))
                
                # Sort by similarity and get top_k
                similarities.sort(reverse=True)
                
                results = []
                for similarity, idx in similarities[:top_k]:
                    result = self.stored_metadata[idx].copy()
                    result['similarity_score'] = similarity
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    def _calculate_pairwise_similarities(self, embeddings1: List[List[float]], embeddings2: List[List[float]]) -> List[List[float]]:
        """Calculate pairwise similarities between two sets of embeddings"""
        similarities = []
        
        for emb1 in embeddings1:
            row_similarities = []
            for emb2 in embeddings2:
                similarity = self._cosine_similarity(emb1, emb2)
                row_similarities.append(similarity)
            similarities.append(row_similarities)
        
        return similarities
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            self.logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _analyze_similarities(self, similarities: List[List[float]], texts1: List[str], texts2: List[str]) -> Dict[str, Any]:
        """Analyze similarity matrix"""
        try:
            similarities_flat = [sim for row in similarities for sim in row]
            
            analysis = {
                'average_similarity': np.mean(similarities_flat),
                'max_similarity': np.max(similarities_flat),
                'min_similarity': np.min(similarities_flat),
                'std_similarity': np.std(similarities_flat),
                'similarity_distribution': {
                    'high': len([s for s in similarities_flat if s > 0.8]),
                    'medium': len([s for s in similarities_flat if 0.5 <= s <= 0.8]),
                    'low': len([s for s in similarities_flat if s < 0.5])
                }
            }
            
            # Find best matches
            best_matches = []
            for i, row in enumerate(similarities):
                best_idx = np.argmax(row)
                best_score = row[best_idx]
                best_matches.append({
                    'text1_index': i,
                    'text2_index': best_idx,
                    'text1': texts1[i] if i < len(texts1) else '',
                    'text2': texts2[best_idx] if best_idx < len(texts2) else '',
                    'similarity': best_score
                })
            
            analysis['best_matches'] = best_matches
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Similarity analysis failed: {e}")
            return {'error': str(e)}
    
    def _perform_clustering(self, n_clusters: int) -> Dict[str, Any]:
        """Perform clustering on stored embeddings"""
        try:
            if len(self.stored_embeddings) < n_clusters:
                return {
                    'success': False,
                    'error': f"Not enough embeddings ({len(self.stored_embeddings)}) for {n_clusters} clusters",
                    'confidence': 0.0
                }
            
            # Simple k-means clustering using numpy
            embeddings_array = np.array(self.stored_embeddings)
            
            # Initialize centroids randomly
            centroids = embeddings_array[np.random.choice(len(embeddings_array), n_clusters, replace=False)]
            
            # Perform k-means iterations
            for _ in range(10):  # Max 10 iterations
                # Assign points to clusters
                distances = np.linalg.norm(embeddings_array[:, np.newaxis] - centroids, axis=2)
                cluster_assignments = np.argmin(distances, axis=1)
                
                # Update centroids
                new_centroids = []
                for i in range(n_clusters):
                    cluster_points = embeddings_array[cluster_assignments == i]
                    if len(cluster_points) > 0:
                        new_centroids.append(np.mean(cluster_points, axis=0))
                    else:
                        new_centroids.append(centroids[i])  # Keep old centroid if no points
                
                centroids = np.array(new_centroids)
            
            # Create cluster results
            clusters = {}
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_assignments == i)[0]
                clusters[f'cluster_{i}'] = {
                    'indices': cluster_indices.tolist(),
                    'texts': [self.stored_metadata[idx]['text'] for idx in cluster_indices if idx < len(self.stored_metadata)],
                    'size': len(cluster_indices),
                    'centroid': centroids[i].tolist()
                }
            
            return {
                'success': True,
                'clusters': clusters,
                'n_clusters': n_clusters,
                'total_points': len(self.stored_embeddings),
                'method': 'kmeans',
                'confidence': 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }
