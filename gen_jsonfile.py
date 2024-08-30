import os
import json
import sys


def gen_json_file(input_dir, output_dir):
    metadata = {}

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            metadata[filename] = {"languages": ["Chinese", "English"]}

    output_path = os.path.join(output_dir, "metadata.json")
    with open(output_path, "w", encoding='utf-8') as json_file:
        json.dump(metadata, json_file, ensure_ascii=False, indent=4)


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_jsonfile.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]

    if not os.path.isdir(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        sys.exit(1)

    gen_json_file(input_dir, input_dir)
    print(f"Metadata file has been created in {input_dir}")


if __name__ == "__main__":
    main()
