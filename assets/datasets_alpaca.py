from datasets import load_dataset
import json

def main():
    # Load the Alpaca dataset from Hugging Face
    print("📥 Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Print basic info
    print(f"\n✅ Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}\n")

    # Show the first few examples
    num_examples = 500
    print(f"🔍 Showing first {num_examples} examples:\n")
    for i in range(num_examples):
        row = dataset[i]
        print(f"Example #{i+1}")
        print(f"Instruction: {row['instruction']}")
        print(f"Input: {row['input']}")
        print(f"Output: {row['output']}")
        print("-" * 80)

    # Optionally save a few examples to a local JSON file
    sample_path = "alpaca_samples.json"
    samples = [dataset[i] for i in range(num_examples)]
    with open(sample_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n💾 Saved first {num_examples} examples to {sample_path}")

if __name__ == "__main__":
    main()

