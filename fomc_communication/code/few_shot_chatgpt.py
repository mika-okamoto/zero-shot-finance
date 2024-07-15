import openai,os,sys
import pandas as pd
from time import sleep, time
from datetime import date
today = date.today()

for seed in [5768, 78516, 944601]:  
    for data_category in ["lab-manual-split-combine"]:
        start_t = time()
        # load training data
        test_data_path = "../data/test/" + data_category + "-test" + "-" + str(seed) + ".xlsx"
        data_df = pd.read_excel(test_data_path)
        train_data_path = "../data/train/" + data_category + "-train" + "-" + str(seed) + ".xlsx"
        train_df = pd.read_excel(train_data_path)

        sample = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 3)))
        few_shot_ex = ""
        d = {0: 'dovish', 1: 'hawkisk', 2: 'neutral'}
        for index, row in sample.iterrows():
            few_shot_ex += f'Sentence: {row.sentence}\nLabel: {d[row.label]}\n\n'
        print(few_shot_ex)

        sentences = data_df['sentence'].to_list()
        labels = data_df['label'].to_numpy()

        output_list = []
        for i in range(len(sentences)): 
            sen = sentences[i]
            message = f"Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEUTRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the monetary policy, 'DOVISH' if it is corresponding to easing of the monetary policy, or 'NEUTRAL' if the stance is neutral. Examples:\n{few_shot_ex}\nProvide the label in the first line and provide a short explanation in the second line. The sentence: " + sen
            # print(message)
            prompt_json = [
                    {"role": "user", "content": message},
            ]
            try:
                chat_completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=prompt_json,
                        temperature=0.0,
                        max_tokens=1000
                )
                answer = chat_completion.choices[0].message.content
            except Exception as e:
                print(e)
                # i = i - 1
                # sleep(10.0)
                answer = "Unsure. Error."

            # print(answer)
            output_list.append([labels[i], sen, answer])
            # sleep(1.0) 
            print(i)

            results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])

            results.to_csv(f'../data/llm_prompt_outputs/few_shot_3_chatgpt_{data_category}_{seed}_{today.strftime("%d_%m_%Y")}.csv', index=False)
            