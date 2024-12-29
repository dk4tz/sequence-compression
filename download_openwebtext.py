import os
from tqdm import tqdm
from datasets import load_dataset

# Constants
TARGET_SIZE_GB = 8
BYTES_PER_GB = 1024 * 1024 * 1024
OUTPUT_FILE = "data/sentences.txt"


def main():
    # Ensure the output directory exists
    os.makedirs("data", exist_ok=True)

    # Calculate the target byte size
    target_bytes = TARGET_SIZE_GB * BYTES_PER_GB
    total_bytes = 0
    count = 0

    print("Streaming OpenWebText dataset from Hugging Face...")
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    # Open the output file for writing
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Processing dataset"):
            # Stop processing if the target size is reached
            if total_bytes >= target_bytes:
                print(f"\nReached target size of {TARGET_SIZE_GB}GB")
                break

            try:
                # Extract text from the current dataset item
                text = item["text"]
                if text:
                    # Write each line of text to the file
                    for line in text.splitlines():
                        line = line.strip()
                        if line:
                            # Convert the line to bytes to calculate its size
                            line_bytes = len(line.encode("utf-8"))
                            if total_bytes + line_bytes > target_bytes:
                                break

                            # Write the line to the file and update the byte counter
                            f.write(line + "\n")
                            total_bytes += line_bytes
                            count += 1

                            # Periodically print progress
                            if count % 1e6 == 0:
                                gb_so_far = total_bytes / BYTES_PER_GB
                                print(
                                    f"\nProcessed {count} sequences ({gb_so_far:.4f} GB)"
                                )
                                print(f"Sample: {line}")

            except Exception as e:
                print(f"Error processing an item: {e}")
                continue

    # Print the final dataset statistics
    final_gb = total_bytes / BYTES_PER_GB
    print(f"\nFinal dataset: {count} sequences, {final_gb:.4f} GB")


if __name__ == "__main__":
    main()
