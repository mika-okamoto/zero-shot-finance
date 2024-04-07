import numpy as np
import pandas as pd
import os
from nltk import word_tokenize

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def decode(x):
    try:
        list_words = word_tokenize(x)
        label_word = list_words[0].lower()
        if (label_word == 'label' or label_word == 'class'): label_word = list_words[2].lower()
        if "dovish" in label_word:
            return 0
        elif "hawkish" in label_word:
            return 1
        elif "neutral" in label_word:
            return 2
        else: 
            print(list_words)
            return -1
    except:
        return -1


acc_list = []
f1_list = []
missing_perc_list = [] 

files = os.listdir('../data/llm_prompt_outputs')

files_xls = ['chatgpt_lab-manual-split-combine_5768_24_02_2024_7.csv', 
             'chatgpt_lab-manual-split-combine_78516_24_02_2024_5.csv',
             'chatgpt_lab-manual-split-combine_944601_24_02_2024_1.csv', 
             'few_shot_3_chatgpt_lab-manual-split-combine_5768_07_04_2024.csv']

for file in files_xls:
    df = pd.read_csv('../data/llm_prompt_outputs/' + file)

    df["predicted_label"] = df["text_output"].apply(lambda x: decode(x))
    acc_list.append(accuracy_score(df["true_label"], df["predicted_label"]))
    f1_list.append(f1_score(df["true_label"], df["predicted_label"], average='weighted'))
    print(f1_list)
    missing_perc_list.append((len(df[df["predicted_label"]==-1])/df.shape[0])*100.0)

print("f1 score mean: ", format(np.mean(f1_list), '.6f'))
print("f1 score std: ", format(np.std(f1_list), '.6f'))
print("Percentage of cases when didn't follow instruction: ", format(np.mean(missing_perc_list), '.6f'), "\n")
