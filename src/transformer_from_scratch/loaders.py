from datasets import load_dataset
import tiktoken 

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
encoder = tiktoken.get_encoding("r50k_base")

print(dataset)

print("\n--- First 5 lines of Training Data ---")
for i in range(5):
    # The data is stored in a list of strings called 'text'
    line = dataset['train']['text'][i]
    print(f"Line {i}: {repr(line)}")

# Join all the rows with a newline character
full_text = "\n".join(dataset['train']['text'])


print(f"First 500 characters:\n{full_text[:500]}")
print(f"\nTotal characters: {len(full_text):,}")
print(f"Length in tokens {len(encoder.encode(full_text)):,}")




