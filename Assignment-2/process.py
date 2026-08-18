import pyarrow.parquet as pq

parquet_file = "te_part_0000.parquet"
txt_file = "train.txt"
target_chars = int(25_000_000 * 0.4)  # 30% of 25M = 7,500,000 chars

# Generator to read text column in batches
def parquet_text_generator(file_path, column="text"):
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches():
        table = batch.to_pandas()
        for text in table[column].astype(str):
            yield text

# Accumulate characters
char_buffer = ""
for text in parquet_text_generator(parquet_file):
    char_buffer += text
    if len(char_buffer) >= target_chars:
        char_buffer = char_buffer[:target_chars]
        break

print(f"Extracted {len(char_buffer)} characters from {parquet_file}")

# Append to train.txt
with open(txt_file, "a", encoding="utf-8") as f:
    f.write(char_buffer)

print(f"Appended to {txt_file}")
