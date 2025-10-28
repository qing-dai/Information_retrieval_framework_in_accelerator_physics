import json
import os

def chunk_paragraphs(text, min_tokens = 120):
    """
    1. Split text on each newline into “paragraphs.”
    2. Always append the next paragraph to the buffer.
    3. As soon as the buffer’s total token count ≥ min_tokens, flush it as one chunk.
    4. At the end, flush any leftover buffer.
    """
    
    # 1) every line is a paragraph
    paras = [line.strip() for line in text.split("\n") if line.strip()]

    chunks = []
    buffer = []

    def flush():
        if buffer:
            chunks.append(" ".join(buffer))
            buffer.clear()

    for para in paras:
        # 2) always buffer the paragraph
        buffer.append(para)
        # 3) check the buffer size
        total_tokens = len(" ".join(buffer).split())
        if total_tokens >= min_tokens:
            flush()
    # 4) flush any remaining paragraphs in the buffer
    flush()
    return chunks

def evaluate_min_tokens(all_text, candidates):
    stats = {}
    for mt in candidates:
        sizes = []
        for item in all_text:
            for chunk in chunk_paragraphs(item["text_data"], min_tokens=mt):
                sizes.append(len(chunk.split()))
        avg = sum(sizes)/len(sizes) if sizes else 0
        stats[mt] = (avg, len(sizes))
    return stats

if __name__ == "__main__":
    # Example usage
    path = "/home/qdai/expert_pdf_mineru/expert_text.json"
    IPAC_2023_json_path = "/home/qdai/IPAC_2023/IPAC_2023/IPAC_2023.json"
    IPAC_2024_json_path = "/home/qdai/IPAC_2024/IPAC_2024.json"
    with open(IPAC_2024_json_path, "r", encoding="utf-8") as f:
        all_text = json.load(f)
    # pick a range of min_tokens to try
    candidates = list(range(100, 301, 50))  # 50,100,150,…,350
    results = evaluate_min_tokens(all_text, candidates)

    # print out: min_tokens → (avg_chunk_size, total_chunks)
    for mt, (avg_size, count) in results.items():
        print(f"min_tokens={mt:3d} → avg {avg_size:.1f} tokens over {count} chunks")
