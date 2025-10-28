# count number of entries in a JSON file
import json
from pathlib import Path
from typing import Any

file = Path("vs_prompt_cache_integer_confidence2.json")  # adjust the path as needed

with file.open("r", encoding="utf-8") as f:
    data: dict[str, Any] = json.load(f)
    print(f"Total entries in {file}: {len(data)}")