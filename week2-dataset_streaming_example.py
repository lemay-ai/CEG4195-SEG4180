# container: scipy
# needs: network
from datasets import load_dataset

# Stream C4, a cleaned Common Crawl (hundreds of GB): no download up front
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
print(next(iter(dataset)))

# Save the first 1000 samples to a local file, reading only as far as needed
with open("sample_data.txt", "w") as f:
    for sample in dataset.take(1000):
        f.write(sample["text"] + "\n")
print("Downloaded 1000 samples and saved to 'sample_data.txt'")
