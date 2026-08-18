with open("input_data.txt", "r", encoding="utf-8") as f:
    data = f.read()
print("encrypt data:",len(data))

with open("train_cleaned.txt", "r", encoding="utf-8") as f:
    data = f.read()
print("Train data:",len(data))