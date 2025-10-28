# models.py
import torch
from torch import nn
from sentence_transformers import SentenceTransformer, models as st_models
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import robust_read_data, build_ir_index
from scorer import encode_pairs, compute_sims

class QwenReranker:
    def __init__(self, name_or_path: str, device: str = "cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token # CHANGED: define pad token
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = "flash_attention_2"
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs).to(device).eval()

         # CHANGED: use space-prefixed tokens for proper matching
        self.token_true_id  = self.tokenizer(" yes", add_special_tokens=False).input_ids[-1]
        self.token_false_id = self.tokenizer(" no",  add_special_tokens=False).input_ids[-1]
        self.max_length = 8192

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    @staticmethod
    def format_instruction(query, doc):
        instruction = 'Given a search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs, max_length=None):
        """
        Build model inputs:
        - tokenize without padding
        - wrap with prefix/suffix
        - append ONE space token so the last row is the next-token position
        - pad to batch-longest with an attention mask
        """
        if max_length is None:
            max_length = self.max_length

        tokenizer = self.tokenizer
        model = self.model

        # tokenization without padding (we add our own later)
        inputs = tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=True,
            max_length=max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        # prefix + content + suffix
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

        # append ONE space token so the very last row corresponds to "next token after suffix"
        space_id = tokenizer(" ", add_special_tokens=False).input_ids[-1]
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i].append(space_id)

        # pad to batch-longest (saves VRAM vs padding to self.max_length)
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")

        # move to device
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)

        return inputs

    @torch.no_grad()
    def score(self, pairs, batch_size=8):
        """
        Return P(yes) for each (query,doc) pair.
        Reads logits at the true next-token row (thanks to the appended space).
        """
        scores = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            inputs = self._process_inputs(chunk)
            logits = self.model(**inputs).logits        # (B, T, V)
            last = logits[:, -1, :]                     # next-token distribution

            # If you configured variant sets (YES_IDS/NO_IDS), use them; else single-token ids.
            if hasattr(self, "YES_IDS") and hasattr(self, "NO_IDS") and len(self.YES_IDS) > 0 and len(self.NO_IDS) > 0:
                yes = last[:, self.YES_IDS].logsumexp(dim=1)
                no  = last[:, self.NO_IDS].logsumexp(dim=1)
            else:
                yes = last[:, self.token_true_id]
                no  = last[:, self.token_false_id]

            two = torch.stack([no, yes], dim=1)         # [P(no), P(yes)] in logit space
            prob_yes = torch.softmax(two, dim=1)[:, 1]

            scores.extend(prob_yes.float().cpu().tolist())

        return scores

def load_model(path, kind, device):
    if kind == "qwen":
        m = SentenceTransformer(
            path, device=device,
            trust_remote_code=True, 
            model_kwargs={"torch_dtype":"float16" if torch.cuda.is_available() else "auto",
                          "attn_implementation":"flash_attention_2" if torch.cuda.is_available() else "sdpa"},
            tokenizer_kwargs={"padding_side":"left"},
        )
        return m
    if kind == "qwen-reranker":
        return QwenReranker(path, device=device)
    # default ST
    m = SentenceTransformer(path, device=device,
                            tokenizer_kwargs={"padding_side":"right","truncation":True})
    return m

if __name__ == "__main__":
    qwen3_small_emb = "/home/qdai/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    model = load_model(qwen3_small_emb, "qwen", "cuda:6", 1024)

    queries = [
    "What is the capital of China?",
    "Explain gravity",
]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    query_emb = model.encode(queries, prompt_name="query")
    document_emb = model.encode(documents)
    sim = model.similarity(query_emb, document_emb)

        
    # query_embeddings, d_embeddings = encode_pairs(model, queries, documents, add_instruction=True, bs=4)
    # sim = compute_sims(query_embeddings, d_embeddings)
    print(sim)
        
    
    

    # reranker = QwenReranker(
    #     "/home/qdai/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3",
    #     "cuda:6"
    # )

    # # path = "/home/qdai/IR_system_test/ir_eval/embedding_reranker_results/test_accphys_top50_reranked.xlsx"
    # # df = pd.read_excel(path)

    # qids = ["Q0", "Q1"]
    # dids = ["D0", "D1", "D2", "D3"]

    # queries = [
    #     # "What is the capital of China?",
    #     # "What is the capital of China?",
    #     "Explain gravity.",
    #     "Explain gravity."
        
    # ]
    # docs = [
    #     # "Beijing is the capital of China.",          # D0
    #     # "Shanghai is a large city in China.",        # D1
    #     "Gravity pulls objects toward each other.",  # D2
    #     "The Moon orbits the Earth."                 # D3
    # ]

    # allowlist = {'Q0': [0, 1], 'Q1': [2, 3]}

    # pairs = [reranker.format_instruction(q, d) for q, d in zip(queries, docs)]
    # print(f"pairs: {pairs}")
    
    # scores = reranker.score(pairs, batch_size=4)

    # print(scores)