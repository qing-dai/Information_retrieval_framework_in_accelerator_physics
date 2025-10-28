import numpy as np
import faiss
import pickle
import os
import random
from pathlib import Path

# Function to normalize a vector so its L2 norm is 1.
def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

# Document class that holds the chunk text, its normalized embedding, and metadata (including filename)
class Document:
    def __init__(self, text, embedding, filename):
        self.text = text
        self.embedding = embedding  # normalized embedding (numpy array)
        self.metadata = {'filename': filename}



# Define a simple vector store using FAISS with inner product (cosine similarity for normalized vectors).
class VectorStore:
    def __init__(self, embedding_dim, use_gpu=False, gpu_id=3):
        self.embedding_dim = embedding_dim
        # Use inner product (IP) because for normalized vectors, IP is equivalent to cosine similarity.
        if use_gpu:
            print(f"Using GPU {gpu_id} for FAISS index")
            # 1) allocate GPU resources
            self.res = faiss.StandardGpuResources()
            # 2) build a CPU flat-ip index
            cpu_index = faiss.IndexFlatIP(embedding_dim)
            # 3) move it to GPU
            self.index = faiss.index_cpu_to_gpu(self.res, gpu_id, cpu_index)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents = []  # Keeps Document objects with metadata
        self.file_to_indices = {}

    def add_documents(self, docs):
        # Extract embeddings from documents and add them to the FAISS index.
        embeddings = np.array([doc.embedding for doc in docs]).astype('float32')
        self.index.add(embeddings)

        # record Document objects and update mapping
        base_idx = len(self.documents)
        for i, doc in enumerate(docs):
            idx = base_idx + i
            fname = doc.metadata['filename']
            # add index to the list of each document object
            self.file_to_indices.setdefault(fname, []). append(idx)
            self.documents.append(doc)

    def search_with_context(self, query_embedding, top_k=5, context = 1):
        '''
        return top_k hits + adjacent {context} chunks in the same file
        '''
        query_embedding = normalize(query_embedding)
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        out = []
        for hit_rank, doc_idx in enumerate(indices[0], start=1):
            if doc_idx >= len(self.documents):
                continue
            doc = self.documents[doc_idx]
            fname = doc.metadata['filename']

            # get the doc_idx from the file_to_indices mapping
            file_list = self.file_to_indices[fname]
            pos = file_list.index(doc_idx)

            # get the context chunks
            ctx_chunks = ''
            for offset in range(-context, context+1):
                neigh_pos = pos + offset
                if 0 <= neigh_pos < len(file_list):
                    neigh_idx = file_list[neigh_pos]
                    ctx_doc = self.documents[neigh_idx]
                    ctx_chunks +=  ctx_doc.text

            out.append({
                "rank": hit_rank,
                "score": float(distances[0][hit_rank-1]),
                'context': ctx_chunks,
                'filename': fname,
            })
        return out


    def search(self, query_embedding, top_k=5):
        # Normalize the query embedding too!
        query_embedding = normalize(query_embedding)
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # Note: higher inner product means higher similarity for normalized vectors.
                results.append({
                    'text': doc.text,
                    'filename': doc.metadata['filename'],
                    'score': distances[0][i]
                })
        return results

    def sample_random(self, query_embedding, top_k=5, sample_k=15):
        """
        Return a list of `sample_k` randomly chosen documents not in the top `top_k` for the query.
        Each result is a dict matching the `search` output format.
        """
        # Normalize and find top_k indices
        q_emb = normalize(query_embedding)
        q_arr = np.array([q_emb]).astype('float32')
        _, indices = self.index.search(q_arr, top_k)
        top_idxs = set(indices[0].tolist())

        # Build candidates excluding top_k
        candidates = [(i, doc) for i, doc in enumerate(self.documents) if i not in top_idxs]
        if not candidates:
            return []
        sampled = random.sample(candidates, min(sample_k, len(candidates)))

        # Compute score and format
        results = []
        for idx, doc in sampled:
            score = float(np.dot(q_emb, doc.embedding))
            results.append({
                'text': doc.text,
                'filename': doc.metadata['filename'],
                'score': score
            })
        return results

    def save(self, index_file , metadata_file):
        # save Faiss index to disk
        index_file = Path(index_file)
        metadata_file = Path(metadata_file)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # 1) If this is a GPU index, convert it to CPU first
        try:
            # index_gpu_to_cpu exists once you install faiss-gpu
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        except AttributeError:
            # not a GPU index (or faiss-gpu not installed), assume it's already CPU
            cpu_index = self.index

        # 2) Now serialize the CPU index
        faiss.write_index(cpu_index, str(index_file))

        # 3) Dump metadata as before
        with open(metadata_file, 'wb') as f:
            pickle.dump((self.documents, self.file_to_indices),
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"VectorStore saved to disk:\n - {index_file}\n - {metadata_file}")

    @classmethod
    def load(cls, embedding_dim, index_file, metadata_file):
        """
        Load FAISS index + metadata. Accepts str or Path.
        """
        index_file = Path(index_file)
        metadata_file = Path(metadata_file)
        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing files:\n - {index_file}\n - {metadata_file}")

        vs = cls(embedding_dim)
        vs.index = faiss.read_index(str(index_file))  # cast to str
        with open(metadata_file, 'rb') as f:
            vs.documents, vs.file_to_indices = pickle.load(f)
        print("VectorStore loaded from disk")
        return vs

# ----- Example Usage -----
