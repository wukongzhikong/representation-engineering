from datasets import load_dataset
import numpy as np
from pandas import DataFrame
from repe_eval.tasks.utils import shuffle_all_train_choices

def medmcqa_dataset(ntrain=25, seed=1):
    template_str = "Consider the correctness of the answer to the following medical question:\n" \
    "Question: {question}\n" \
    "Answer: {answer}.\n" \
    "The probability the answer being correct is "

    def clean_answer_s(s):
        return s[:-1] if s[-1] == "." else s
    
    def format_samples(row):
        prompts = []

        question = row.question
        correct_option_index = row.cop
        options = ['opa', 'opb', 'opc', 'opd']
        correct_option = options[correct_option_index]
        correct_answer = getattr(row, correct_option)

        prompts.append(template_str.format(question=question, answer=correct_answer))
        
        for option in options:
            if option != correct_option:
                answer = getattr(row, option)
                prompts.append(template_str.format(question=question, answer=answer))    

        return prompts, [1] + [0] * (len(options) - 1) 

    def _keep_4_options_row(e):
        return len(e['choices']['label']) == 4

    def samples(df: DataFrame):
        prompts, labels = [], []
        for row in df.itertuples():
            answer_prompts, label =  format_samples(row)
            prompts.append(answer_prompts)
            labels.append(label)
        return prompts, labels

    dataset = load_dataset("/home2/yhn/data/datasets/medmcqa")
    train_df = dataset['train'].shuffle(seed=seed).to_pandas()
    test_df = dataset['test'].to_pandas()
    val_df = dataset['validation'].to_pandas()

    train_data, train_labels = samples(train_df)
    test_data, test_labels = samples(test_df)
    val_data, val_labels = samples(val_df)


    train_data, train_labels =  train_data[:ntrain], train_labels[:ntrain]
    train_data, train_labels = shuffle_all_train_choices(train_data, train_labels, seed)

    train_data =  np.concatenate(train_data).tolist()
    test_data =  np.concatenate(test_data).tolist()
    val_data = np.concatenate(val_data).tolist()

    return {
            "train": {"data": train_data, "labels": train_labels}, 
            "test": {"data": test_data, "labels": test_labels}, 
            "val": {"data": val_data, "labels": val_labels}
            }

def test_load_medmcqa():
    medmcqa_dataset()

if __name__ == "__main__":
    test_load_medmcqa()