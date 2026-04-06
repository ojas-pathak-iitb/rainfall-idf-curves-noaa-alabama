import os
import tarfile
import unlzw3

# --- Paths (relative to project root) ---
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER     = os.path.join(BASE_DIR, "Dataset", "state_data", "01")
OUTPUT_FOLDER    = os.path.join(BASE_DIR, "Dataset", "extracted")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Extract all .tar.Z files ---
files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tar.Z")])

if not files:
    print(f"No .tar.Z files found in: {INPUT_FOLDER}")
    print("Make sure the Dataset/state_data/01/ folder contains the Alabama files.")
else:
    print(f"Found {len(files)} file(s) to extract.\n")
    for filename in files:
        filepath = os.path.join(INPUT_FOLDER, filename)
        print(f"Processing: {filename} ...", end=" ", flush=True)

        # Step 1: Read and decompress the .Z file
        with open(filepath, "rb") as f:
            compressed = f.read()

        tar_bytes = unlzw3.unlzw(compressed)

        # Step 2: Write to a temporary .tar file
        temp_tar = os.path.join(OUTPUT_FOLDER, filename.replace(".tar.Z", ".tar"))
        with open(temp_tar, "wb") as f:
            f.write(tar_bytes)

        # Step 3: Extract the .tar to the output folder
        with tarfile.open(temp_tar) as tar:
            tar.extractall(path=OUTPUT_FOLDER)

        # Step 4: Remove the temporary .tar file
        os.remove(temp_tar)
        print("Done.")

    print(f"\nAll files extracted successfully.")
    print(f"Extracted files are in: {OUTPUT_FOLDER}")