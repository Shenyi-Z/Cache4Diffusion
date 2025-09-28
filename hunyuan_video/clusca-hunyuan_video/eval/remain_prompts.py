import os
import json
from collections import defaultdict

# Load VBench_full_info.json
json_file = "VBench_full_info.json"
with open(json_file, "r", encoding="utf-8") as f:
    vbench_data = json.load(f)

# Get the video filenames in the original folder
original_folder = "/root/autodl-tmp/HunyuanVideo_samples/VBench_Results/original/gen_videos"
existing_videos = defaultdict(set)  # Store which indices have been generated for each prompt_en

# Iterate through the original folder and collect generated files
for filename in os.listdir(original_folder):
    if filename.endswith(".mp4"):  # Consider only video files
        parts = filename.rsplit("-", 1)  # Split into prompt and index
        if len(parts) == 2 and parts[1][0].isdigit():  # Ensure correct format
            prompt_part, index = parts
            existing_videos[prompt_part].add(index.split(".")[0])  # Store index (remove .mp4)

# Filter out JSON entries for videos that are not fully generated
remaining_entries = []
for idx, entry in enumerate(vbench_data):
    prompt_text = entry["prompt_en"]
    required_indices = {"0", "1", "2", "3", "4"}  # 5 videos are required
    if prompt_text not in existing_videos or existing_videos[prompt_text] != required_indices:
        remaining_entries.append(entry)

# Count and print the number of remaining prompts
remaining_count = len(remaining_entries)
print(f"Total remaining prompts: {remaining_count}")

# Generate VBench_remain_info.json
output_file = "VBench_remain_info.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(remaining_entries, f, indent=4, ensure_ascii=False)

print(f"Processing complete. Ungenerated or incomplete video information has been saved in {output_file}")
