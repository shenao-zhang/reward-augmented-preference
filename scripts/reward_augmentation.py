from datasets import load_dataset, DatasetDict, concatenate_datasets
USERNAME = "ZhangShenao"

def system_prompt(overall_score):
    overall_score = int(overall_score) if int(overall_score) == overall_score else overall_score
    system_content = \
f"""You are an assistant that generates responses for the instruction while implicitly achieving the following target scores (on a scale of 1-10, where 1 is lowest and 10 is highest):
Overall score: {overall_score}."""
    return system_content

def add_pos_goal(example):
    overall_score = example["overall_score_chosen"]
    system_msg = {"content": system_prompt(overall_score), "role": "system"}
    example["chosen"].insert(0, system_msg)
    example["rejected"].insert(0, system_msg)
    return example

def add_neg_goal(example):
    overall_score = example["overall_score_rejected"]
    system_msg = {"content": system_prompt(overall_score), "role": "system"}
    gc_chosen = [system_msg] + example["rejected"]
    gc_rejected = [system_msg] + example["chosen"]
    example["chosen"] = gc_chosen
    example["rejected"] = gc_rejected
    return example

ds = load_dataset(f"{USERNAME}/ultrafeedback_preprocess", split="train_prefs")
ds = ds.filter(lambda x: x["avg_fine_score_chosen"] >= x["avg_fine_score_rejected"], num_proc=8)
pos_gc_ds = ds.map(add_pos_goal)
neg_gc_ds = ds.map(add_neg_goal)

all_ds = DatasetDict()
all_ds["train_prefs"] = concatenate_datasets([pos_gc_ds, neg_gc_ds])
all_ds["test_prefs"] = load_dataset(f"{USERNAME}/ultrafeedback_preprocess", split="test_prefs")
all_ds.push_to_hub(f"{USERNAME}/ultrafeedback_reward_augmented", private=True)

def add_pos_goal_nosys(example):
    overall_score = example["overall_score_chosen"]
    system_msg = system_prompt(overall_score) + "\n\n"
    example["chosen"][0]["content"] = system_msg + example["chosen"][0]["content"]
    example["rejected"][0]["content"] = system_msg + example["rejected"][0]["content"]
    return example

def add_neg_goal_nosys(example):
    overall_score = example["overall_score_rejected"]
    system_msg = system_prompt(overall_score) + "\n\n"
    gc_chosen = example["rejected"][0]
    gc_rejected = example["chosen"][0]
    example["chosen"][0]["content"] = system_msg + gc_chosen["content"]
    example["rejected"][0]["content"] = system_msg + gc_rejected["content"]
    return example

pos_gc_ds = ds.map(add_pos_goal_nosys)
neg_gc_ds = ds.map(add_neg_goal_nosys)

all_ds = DatasetDict()
all_ds["train_prefs"] = concatenate_datasets([pos_gc_ds, neg_gc_ds])
all_ds["test_prefs"] = load_dataset(f"{USERNAME}/ultrafeedback_preprocess", split="test_prefs")
all_ds.push_to_hub(f"{USERNAME}/ultrafeedback_reward_augmented_nosys", private=True)