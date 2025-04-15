import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


medqa = load_dataset('/home2/yhn/data/datasets/medmcqa')['validation']
model_path_list = [
    # medical models to validate honesty: Bio-Medical-Llama-3-8B, JSL-MedLlama-3-8B-v2.0, Medical-Llama3-v2
    "/home2/yhn/data/models/huggingface/ContactDoctor/Bio-Medical-Llama-3-8B",
    "/home2/yhn/data/models/huggingface/johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
    "/home2/yhn/data/models/huggingface/ruslanmv/Medical-Llama3-v2",
]
for model_path in model_path_list:
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", return_dict_in_generate=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with torch.no_grad():
        correct = 0
        total = len(medqa)
        for item in tqdm(medqa):
            question = item["question"]
            options_to_answers = {'A': item['opa'],
                                  'B': item['opb'],
                                  'C': item['opc'],
                                  'D': item['opd']}
            correct_option = chr(item["cop"] + ord('A'))
            scores = {}

            prompt0 = "".join(["Question: ", question])
            prompt0 += "\nOptions:\n"
            for (option, answer) in options_to_answers.items():
                prompt0 += option
                prompt0 += ". "
                prompt0 += answer
                prompt0 += "\n"
            for (option, answer) in options_to_answers.items():
                prompt = "".join([prompt0, 'Answer: ', option, '. ', answer])
                input_ids = tokenizer(prompt0, return_tensors="pt").input_ids.to(model.device)
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

                outputs = model(prompt_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[input_ids.shape[-1] - 1: -1, :]
                prompt_ids = prompt_ids[0, input_ids.shape[-1]:]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                log_probs = log_probs[3:]  # drop the '\nAnswer:' prefix

                scores[option] = log_probs.sum().item()
            
            if scores[correct_option] == max(scores.values()):
                correct += 1
        acc = f"{correct/total*100:.1f}%"
        print(correct, total, acc)