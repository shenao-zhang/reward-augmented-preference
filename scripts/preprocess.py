from datasets import load_dataset, DatasetDict, concatenate_datasets
import hashlib
import random
import time
USERNAME = "ZhangShenao"
random.seed(42)

# Load revision with the fixes to overall_score
ds = load_dataset("openbmb/UltraFeedback", split="train", revision="40b436560ca83a8dba36114c22ab3c66e43f6d5e")

# Load TrutfulQA prompts to ensure we remove samples from evol_instruct
tqa_a = load_dataset("truthful_qa", "generation", split="validation")
tqa_b = load_dataset("truthful_qa", "multiple_choice", split="validation")

total_rows = ds.num_rows

ds = ds.filter(lambda x: x["source"] != "truthful_qa", num_proc=4)
print(f"Remaining samples after removing the TruthfulQA source [{ds.num_rows} / {total_rows}]")

contaminated_prompts = list(set(tqa_a["question"] + tqa_b["question"]))
ds = ds.filter(lambda x: x["instruction"] not in contaminated_prompts, num_proc=4)
print(f"Remaining samples after removing the contaminated prompts [{ds.num_rows} / {total_rows}]")

def get_pairwise_completions(completions):
    start = time.time()
    scores_and_completions = []
    for c in completions:
        helpfulness = c["annotations"]["helpfulness"]["Rating"]
        honesty = c["annotations"]["honesty"]["Rating"]
        instruction_following = c["annotations"]["instruction_following"]["Rating"]
        truthfulness = c["annotations"]["truthfulness"]["Rating"]
        if helpfulness == 'N/A' or honesty == 'N/A' or instruction_following == 'N/A' or truthfulness == 'N/A':
            continue
        scores_and_completions.append((c["overall_score"], c["response"], c["model"],
                                       float(helpfulness), float(honesty), float(instruction_following), float(truthfulness)))
    if len(scores_and_completions) < 2:
        return None, None
    chosen = max(scores_and_completions, key=lambda x: x[0])
    rejected = random.choice(scores_and_completions)
    while rejected == chosen:
        end = time.time()
        if end - start > 3:
            print("Timeout")
            print(chosen, rejected)
            break
        rejected = random.choice(scores_and_completions)
    return chosen, rejected


def format_prompt(x):
    prompt = x["instruction"]
    chosen, rejected = get_pairwise_completions(x["completions"])
    print(chosen, rejected)
    chosen_messages = []
    rejected_messages = []
    chosen_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen[1] if chosen is not None else "N/A"},
    ]
    rejected_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected[1] if rejected is not None else "N/A"},
    ]
    return {
        "prompt": prompt,
        "prompt_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "messages": chosen_messages, # Use best-ranked example for SFT
        "overall_score_chosen": chosen[0] if chosen is not None else -100.0,
        "overall_score_rejected": rejected[0] if rejected is not None else -100.0,
        "fine_grain_score_chosen": chosen[-4:] if chosen is not None else None,
        "fine_grain_score_rejected": rejected[-4:] if rejected is not None else None,
        "avg_fine_score_chosen": sum(chosen[-4:]) / 4 if chosen is not None else None,
        "avg_fine_score_rejected": sum(rejected[-4:]) / 4 if rejected is not None else None,
    }

ds = ds.map(format_prompt, num_proc=8, remove_columns=ds.column_names)


# filter out margin = -100
ds = ds.filter(lambda x: x["overall_score_chosen"] != -100 or x["overall_score_rejected"] != -100, num_proc=8)



def remove_last_step_for_rl(example):
    example["messages"] = example["messages"][:-1]  # remove the assistant response
    return example


all_ds = DatasetDict()

split_dataset = ds.train_test_split(test_size=2000, seed=42, shuffle=True)
test_datasets = split_dataset["test"].train_test_split(0.5, seed=42, shuffle=True)

all_ds["train_prefs"] = split_dataset["train"]
all_ds["train_sft"] = split_dataset["train"]
# Keep more examples for test accuracy
all_ds["test_prefs"] = concatenate_datasets([test_datasets["train"], test_datasets["test"]])
all_ds["test_sft"] = test_datasets["train"]


# remove empty last turns
def filter_empty_messages(example):
    if example["messages"][-1]["role"] == "user":
        example["messages"] = example["messages"][:-1]
    if example["chosen"][-1]["role"] == "user":
        example["chosen"] = example["chosen"][:-1]
    if example["rejected"][-1]["role"] == "user":
        example["rejected"] = example["rejected"][:-1]
    return example


all_ds = all_ds.map(filter_empty_messages)

all_ds["train_gen"] = all_ds["train_sft"].map(remove_last_step_for_rl)
all_ds["test_gen"] = all_ds["test_sft"].map(remove_last_step_for_rl)

assistant_rows = []

# check that gen split does not end with `assistant`, should print 0
for idx, row in enumerate(all_ds["train_gen"]):
    if row["messages"][-1]["role"] == "assistant":
        assistant_rows.append(row)
for row in all_ds["test_gen"]:
    if row["messages"][-1]["role"] == "assistant":
        assistant_rows.append(row)

assert len(assistant_rows) == 0


all_ds.push_to_hub(f"{USERNAME}/ultrafeedback_preprocess", private=True)