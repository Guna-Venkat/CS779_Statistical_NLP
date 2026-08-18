import subprocess
subprocess.run(["pip", "install", "sacrebleu"])

import nltk
from sacrebleu import sentence_chrf
import io
import csv


nltk.download('punkt')

def calculate_chrf_score(file1_path, file2_path):
    file1_data = []
    with open(file1_path, 'r', encoding='utf-8') as f:
        for input_line in f:
            row = [column.strip('"') for column in input_line.split(maxsplit=1)]
            if len(row) >= 2:
                file1_data.append((row[0], nltk.word_tokenize(row[1])))

    file2_data = []
    with open(file2_path, 'r', encoding='utf-8') as f:
        for input_line in f:
            row = [column.strip('"') for column in input_line.split(maxsplit=1)]
            if len(row) >= 2:
                file2_data.append((row[0], nltk.word_tokenize(row[1])))


    if len(file1_data) != len(file2_data):
        raise ValueError(f"Number of data points in file1({len(file1_data)}) and file2({len(file2_data)}) must be the same")     

    chrF_score = []
    totalscore = 0.0
    total_itrs =0
    for (answer_id, hyp), (ref_id, ref) in zip(file1_data, file2_data):
        total_itrs = total_itrs + 1
        if total_itrs==1:
            continue
             
        if answer_id != ref_id:
            print(f"Warning: Answer ID ({answer_id}) and Ref ID ({ref_id}) do not match. Skipping this iteration.")
            continue 
        hyp_text = ' '.join(hyp)
        ref_text = ' '.join(ref)

        if (ref_text and hyp_text):
            scores = sentence_chrf(hyp_text, [ref_text])
            chrF_score.append(scores.score)
            totalscore = totalscore + scores.score

    totalscore = totalscore / 100    
    return (totalscore/total_itrs)

if __name__ == "__main__":
    file1_path = 'truth.csv'  # Replace with the actual path to file1
    file2_path = 'answer.csv'  # Replace with the actual path to file2 

    score = calculate_chrf_score(file1_path, file2_path)
    print(score)
