import os
from typing import Any, List, Union, Dict
import torch
import pandas as pd
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
import model_utils

class MultiDatasetDataCollatorCompletion(DataCollatorForCompletionOnlyLM):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        is_dpo = examples[0].get("logps", None)
        keys = ["input_ids", "attention_mask"]
        if is_dpo is not None:
            keys.append("logps")

        clean_examples = [{key: ex[key] for key in keys} for ex in examples]
        safe_examples = [
            {key: ex["Safe_Model"][key] for key in keys}
            for ex in examples
            if isinstance(ex["Safe_Model"], dict)
        ]
        clean_examples += safe_examples
        batch = super().torch_call(clean_examples)
        batch["dataset_id"] = torch.concat(
            [
                torch.tensor([ex["dataset_id"] for ex in examples]),
                torch.ones(len(safe_examples)),
            ]  # Hardcoded value of 1 for safe examples
        )

        return batch


def get_dataset_ids():
    return [0, 1, 2]


def get_dataset_id(key):
    dataset_ids = {"away": 0, "toward": 1, "utility": 2}
    return dataset_ids[key]


def get_prompt_formatting_func_and_collator(model_name, tokenizer, collator_type="multi"):
    first_user_msg, user_chat_template, response_template, response_key = model_utils.get_chat_template(
        model_name
    )
    print("DEBUG responsekey raw:", repr(response_key))  # add this
    dataset_text_field, dataset_target_field = get_dataset_text_and_target_field()

    def prompt_formatting_func(sample, input_only=False):
        all_formatted_prompts = []
        for i in range(len(sample[dataset_text_field])):
            formatted_prompt = ""

            # Get system prompt if present, else empty string
            system = sample["System"][i][0] if "System" in sample else ""



            formatted_instruction = first_user_msg.format(system=system, instruction=sample[dataset_text_field][i][0])
            if input_only:
                formatted_prompt = formatted_instruction + response_key
            else:
                formatted_target = response_template.format(target=sample[dataset_target_field][i][0])
                formatted_prompt += formatted_instruction + formatted_target
                for instruction, target in zip(
                    sample[dataset_text_field][i][1:], sample[dataset_target_field][i][1:]
                ):
                    formatted_instruction = user_chat_template.format(instruction=instruction)
                    formatted_target = response_template.format(target=target)
                    formatted_prompt += formatted_instruction + formatted_target

            all_formatted_prompts.append(formatted_prompt)

        return all_formatted_prompts

    # NOTE the phi-3 tokenizer adds the SPIECE_UNDERLINE token (29871) to an encoded <|assistant|> token if not other token is present except for \n, which messes with the matching
    if model_name == "phi-3":
        response_key = [13, 32001, 13]
    if "llama-3" in model_name:
        print("!!!!!!!!!Hello!!!!!!!!")
        print(response_key)
        #response_key_ids = tokenizer.encode(response_key, add_special_tokens=False)
#    if "llama-3" in model_name:
#        response_key = tokenizer.encode(
#            "<|start_header_id|>assistant<|end_header_id|>\n\n",
#            add_special_tokens=False
#        )
    if collator_type == "multi":
        collator = MultiDatasetDataCollatorCompletion(response_key, tokenizer=tokenizer)
    elif collator_type == "single":
        collator = DataCollatorForCompletionOnlyLM(response_key, tokenizer=tokenizer)
    else:
        raise ValueError(f"Collator type {collator_type} not supported")

    return prompt_formatting_func, collator, response_key


DATASETS = [
    "ultrachat",
    "ultrachat_200k",
    "adv_training_behaviors",
    "adv_val_behaviors",
    "adv_test_behaviors",
    "adv_training_safe_prompts",
]


def load_specific_dataset(data_path, dataset_name, split=None, multiple_targets=False):
    if dataset_name not in DATASETS:
        raise NotImplementedError(f"Dataset {dataset_name} not supported, choose from: {DATASETS}")

    if dataset_name == "ultrachat":
        dataset_path = data_path + "utility/ultrachat"
        if os.path.isdir(dataset_path) and len(os.listdir(dataset_path)) > 0:
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset("stingning/ultrachat", split="train")
            dataset.save_to_disk(dataset_path)
        dataset = dataset.map(
            lambda example: {
                "User": example["data"][::2],
                "Model": [ans for ans in example["data"][1::2]],
            }
        )
        frac = int(0.9 * len(dataset))
        dataset = dataset.select(range(frac)), dataset.select(range(frac, len(dataset)))
    if dataset_name == "ultrachat_200k":
        dataset_path_train = data_path + "utility/ultrachat_200k_train"
        if os.path.isdir(dataset_path_train) and len(os.listdir(dataset_path_train)) > 0:
            dataset_train = load_from_disk(dataset_path_train)
        else:
            dataset_train = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            dataset_train.save_to_disk(dataset_path_train)

        dataset_train = dataset_train.map(
            lambda example: {
                "User": [ans["content"] for ans in example["messages"][::2]],
                "Model": [ans["content"] for ans in example["messages"][1::2]],
            }
        )
        # Assuming each token is on average ~4 chars, limit to ~256tokens
        dataset_train = dataset_train.filter(lambda example: len(example["User"][0]) < 768)

        dataset_path_eval = data_path + "utility/ultrachat_200k_eval"
        if os.path.isdir(dataset_path_eval) and len(os.listdir(dataset_path_eval)) > 0:
            dataset_eval = load_from_disk(dataset_path_eval)
        else:
            dataset_eval = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
            dataset_eval.save_to_disk(dataset_path_eval)
        dataset_eval = dataset_eval.map(
            lambda example: {
                "User": [ans["content"] for ans in example["messages"][::2]],
                "Model": [ans["content"] for ans in example["messages"][1::2]],
            }
        )
        # Assuming each token is on average ~4 chars, limit to ~256tokens
        dataset_eval = dataset_eval.filter(lambda example: len(example["User"][0]) < 768)
        dataset = dataset_train, dataset_eval
    elif dataset_name == "adv_training_behaviors":
        train_behavior_filename = (
            data_path + "behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv"
        )
        train_target_filename = data_path + "optimizer_targets/extra_targets/adv_training_targets.json"
        df = create_df_from_behavior_and_target(train_behavior_filename, train_target_filename)
        df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
        dataset = Dataset.from_pandas(df)

        if not multiple_targets:
            dataset = dataset.map(
                lambda batch: {
                    "User": [[batch["User"][0]] for _ in range(len(batch["Model"][0]))],
                    "Model": [[target] for target in batch["Model"][0]],
                },
                batched=True,
                batch_size=1,
            )
        else:
            dataset = dataset.map(
                lambda batch: {
                    "User": [batch["User"]],
                    "Model": batch["Model"],
                },
                batched=True,
                batch_size=1,
            )

    elif dataset_name == "adv_val_behaviors":
        val_behavior_filename = data_path + "behavior_datasets/harmbench_behaviors_text_val.csv"
        val_target_filename = data_path + "optimizer_targets/extra_targets/adv_training_val_targets.json"
        df = create_df_from_behavior_and_target(val_behavior_filename, val_target_filename)
        df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            lambda batch: {
                "User": [batch["User"]],
                "Model": [[batch["Model"][0][0]]],
            },
            batched=True,
            batch_size=1,
        )
    elif dataset_name == "adv_test_behaviors":
        test_behavior_filename = data_path + "behavior_datasets/harmbench_behaviors_text_all.csv"
        test_target_filename = data_path + "optimizer_targets/harmbench_targets_text.json"
        df = create_df_from_behavior_and_target(test_behavior_filename, test_target_filename)
        df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            lambda batch: {
                "User": [batch["User"]],
                "Model": [[batch["Model"][0]]],
            },
            batched=True,
            batch_size=1,
        )
    elif dataset_name == "adv_training_safe_prompts":
        safe_prompts_file_name = data_path + "safe_responses/adv_training_behaviors_safe_responses.csv"
        df = pd.read_csv(safe_prompts_file_name)
        df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
        dataset = Dataset.from_pandas(df)

    return dataset


def create_df_from_behavior_and_target(behavior_path, target_path):
    df_behavior = pd.read_csv(
        behavior_path,
        usecols=["Behavior", "BehaviorID"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    df_target = pd.read_json(target_path, typ="series").reset_index()
    df_target.columns = ["BehaviorID", "Target"]
    df = df_behavior.merge(df_target, on="BehaviorID")
    df = df[["Behavior", "Target"]]
    return df

def create_df_from_path(path):
    df = pd.read_csv(path, encoding="utf-8")
    records = []
    for _, row in df.iterrows():
        toward_targets = [row[f"toward_target_{i}"] for i in range(1,2) if pd.notna(row.get(f"toward_target_{i}", float("nan")))]
        away_targets   = [row[f"away_target_{i}"]   for i in range(1,13) if pd.notna(row.get(f"away_target_{i}", float("nan")))]
        records.append({
            "System": row["system_prompt"],
            "User": row["user_prompt"],
            "Model": away_targets,
            "Safe_Model": toward_targets[0],
        })
#        for i in range(4):
#            records.append({
#                "System": row["system_prompt"],
#                "User": row["user_prompt"],
#                "Model": away_targets,    
#                "Safe_Model": toward_targets[i],  #rotate through all 4
#            })
#        # for away in away_targets:
        #     for toward in toward_targets:
        #         records.append({
        #             "System": row["system_prompt"],
        #             "User": row["user_prompt"],
        #             "Model": away,    
        #             "Safe_Model": toward,  
        #         })
    return pd.DataFrame(records)


def load_adversarial_training_data(
    data_path,
    utility_data,
    probabilities,
    model_name,
    tokenizer,
    stopping_strategy="first_exhausted",
    diverse_safe_answers=False,
    restricted_trainingset_size=None,
):
    # train_behavior_filename = (
    #     data_path + "behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv"
    # )
    # train_target_filename = data_path + "optimizer_targets/extra_targets/adv_training_targets.json"

    # val_behavior_filename = data_path + "behavior_datasets/harmbench_behaviors_text_val.csv"
    # val_target_filename = data_path + "optimizer_targets/extra_targets/adv_training_val_targets.json"

    # df_train = create_df_from_behavior_and_target(train_behavior_filename, train_target_filename)
    # if diverse_safe_answers:
    #     safe_data_path = data_path + "safe_responses/adv_training_behaviors_safe_responses_longer.csv"
    # else:
    #     safe_data_path = data_path + "safe_responses/adv_training_behaviors_safe_responses.csv"

    # df_train_safe_prompts = pd.read_csv(safe_data_path)
    # df_train = pd.merge(df_train, df_train_safe_prompts, on="Behavior")
    # df_train = df_train.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
    
    training_data_path = data_path + "sampled_adv_training_fixed_v2.csv"
    df_train = create_df_from_path(training_data_path)

    # df_val = create_df_from_behavior_and_target(val_behavior_filename, val_target_filename)
    # df_val = df_val.rename(columns={"Behavior": "User", "Target": "Model"})
    # df_val["Safe_Model"] = "Sorry, I cannot do that."

    train_data = Dataset.from_pandas(df_train)
    # eval_data = Dataset.from_pandas(df_val)
    # TODO change eval safe response

    # add dataset ids and duplicate behaviours with different targets
    ## train
    # train_data = train_data.map(
    #     lambda batch: {
    #         "dataset_id": [0 for _ in range(len(batch["Model"][0]))],
    #         "User": [[batch["User"][0]] for _ in range(len(batch["Model"][0]))],
    #         "Model": [[target] for target in batch["Model"][0]],
    #         "Safe_Model": [[batch["Safe_Model"][0]] for _ in range(len(batch["Model"][0]))],
    #     },
    #     batched=True,
    #     batch_size=1,
    # )
    # ## eval
    # eval_data = eval_data.map(
    #     lambda batch: {
    #         "dataset_id": [0 for _ in range(len(batch["Model"][0]))],
    #         "User": [[batch["User"][0]] for _ in range(len(batch["Model"][0]))],
    #         "Model": [[target] for target in batch["Model"][0]],
    #         "Safe_Model": [[batch["Safe_Model"][0]] for _ in range(len(batch["Model"][0]))],
    #     },
    #     batched=True,
    #     batch_size=1,
    # )
    train_data = train_data.map(
        lambda batch: {
            "dataset_id":  [0 for _ in range(len(batch["Model"][0]))],
            "System":      [[batch["System"][0]] for _ in range(len(batch["Model"][0]))],
            "User":        [[batch["User"][0]]   for _ in range(len(batch["Model"][0]))],
            "Model":       [[target]             for target in batch["Model"][0]],
            "Safe_Model":  [batch["Safe_Model"][0] for _ in range(len(batch["Model"][0]))],
        },
        batched=True,
        batch_size=1,
    )

    # Tokenize Safe_Model — same as original but with system added
    first_user_msg, _, response_template, _ = model_utils.get_chat_template(model_name)
    train_data = train_data.map(
        lambda example: {
            "Safe_Model": tokenizer(
                first_user_msg.format(
                    system=example["System"][0],
                    instruction=example["User"][0],
                )
                + response_template.format(target=example["Safe_Model"]),
                max_length=1024,
                truncation=True,
                padding=False,
            ),
        }
    )
#Note example["System"][0] and example["User"][0] — not [0][0] — because after the expansion map, System is ["system_prompt_string"] (a list of one string), not [["system_prompt_string"]]. The [0][0] bug in your current code was another source of subtle issues.
#If you want to use all 4 toward targets as safe responses instead of just toward_targets[0], the cleanest way that stays true to the original structure is to do a second expansion in create_df_from_path — one record per (behavior, toward_target) pair, keeping Model as the full away list each time — but that's optional and the single safe response approach matches the original exactly.
#    train_data = train_data.map(
#        lambda batch: {
#            "dataset_id": [0 for _ in range(len(batch["User"]))],
#            "System":     [[s] for s in batch["System"]],
#            "User":       [[u] for u in batch["User"]],
#            "Model":      [[m] for m in batch["Model"]],
#            "Safe_Model": [s   for s  in batch["Safe_Model"]],  # stays string for tokenization
#        },
#        batched=True,
#    )
#    first_user_msg, user_chat_template, response_template, _ = model_utils.get_chat_template(model_name)
#    train_data = train_data.map(
#        lambda example: {
#            "Safe_Model": tokenizer(
#                first_user_msg.format(
#                    system=example["System"][0][0],       
#                    instruction=example["User"][0][0]
#                )
#                + response_template.format(target=example["Safe_Model"]),
#                max_length=2048,
#                truncation=True,
#                padding=False,
#            ),
#        }
#    )
    # Decode what Safe_Model actually contains
    sample = train_data[0]
    safe_ids = sample["Safe_Model"]["input_ids"]
    decoded = tokenizer.decode(safe_ids)
    print("=== SAFE_MODEL DECODED ===")
    print(repr(decoded))
    # format and tokenize safe response
    # first_user_msg, user_chat_template, response_template, _ = model_utils.get_chat_template(model_name)
    # ## train
    # train_data = train_data.map(
    #     lambda example: {
    #         "Safe_Model": tokenizer(
    #             first_user_msg.format(instruction=example["User"][0])
    #             + response_template.format(target=example["Safe_Model"][0])
    #         ),
    #     }
    # )
    # ## eval
    # eval_data = eval_data.map(
    #     lambda example: {
    #         "Safe_Model": tokenizer(
    #             first_user_msg.format(instruction=example["User"][0])
    #             + response_template.format(target=example["Safe_Model"][0])
    #         ),
    #     }
    # )

    # add utility data
    # Print one sample to verify it looks correct
    sample = train_data[0]
    print("=== SAMPLE CHECK ===")
    print("Keys:", list(sample.keys()))
    print("System:", sample["System"])
    print("User:", sample["User"])
    print("Model:", sample["Model"])
    print("Safe_Model type:", type(sample["Safe_Model"]))
    print("dataset_id:", sample["dataset_id"])
    
    # ===== DIAGNOSTIC — remove after debugging =====
    sample = train_data[0]
    response_key_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    ids_isolated = tokenizer.encode(response_key_str, add_special_tokens=False)
    print("Response key isolated:", ids_isolated)

    full_prompt = (
    "<|start_header_id|>system<|end_header_id|>\n\ntest<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\ntest<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    ids_in_context = tokenizer.encode(full_prompt, add_special_tokens=False)
    print("Response key in context:", ids_in_context[-4:])  # last 4 tokens should be the response key

    print("BOS token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("add_bos_token:", getattr(tokenizer, 'add_bos_token', 'N/A'))

    # Check if response key appears in Safe_Model input_ids
    safe_ids = sample["Safe_Model"]["input_ids"]
    print("Safe_Model input_ids[:10]:", safe_ids[:10])
    print("Safe_Model input_ids[-10:]:", safe_ids[-10:])
    # ===== END DIAGNOSTIC =====
    # Decode what prompt_formatting_func would produce
    from model_utils import get_chat_template
    first_user_msg, _, response_template, _ = get_chat_template(model_name)
    prompt = first_user_msg.format(system=sample["System"][0], instruction=sample["User"][0])
    full = prompt + response_template.format(target=sample["Model"][0])
    print("=== FORMATTED PROMPT ===")
    print(repr(full))
    if utility_data is not None and utility_data != "None":
        utility_train_data, utility_eval_data = load_specific_dataset(data_path, utility_data)
        #### Format train data ####
        utility_train_data = utility_train_data.map(
            lambda example: {"dataset_id": 2, "System": ["You are a helpful, respectful and honest assistant."], "Safe_Model": None, **example}, num_proc=16
        )
        columns = [k for k in utility_train_data.column_names if k not in train_data.column_names]
        utility_train_data = utility_train_data.remove_columns(columns)
        #### Format eval data ####
        utility_eval_data = utility_eval_data.map(
            lambda example: {"dataset_id": 2, "System": ["You are a helpful, respectful and honest assistant."], "Safe_Model": None, **example}, num_proc=16
        )
        columns = [k for k in utility_eval_data.column_names]
        utility_eval_data = utility_eval_data.remove_columns(columns)
        # TODO may want to reload the dataloader at each epoch to get different samples from larger dataset
        train_data = interleave_datasets(
            [train_data, utility_train_data],
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
        )
        # eval_data = interleave_datasets(
        #     [eval_data, utility_eval_data],
        #     probabilities=probabilities,
        #     stopping_strategy=stopping_strategy,
        # )
        eval_data = utility_eval_data.select(range(256))
    else:
        eval_data = None # no utility data
    # After interleave_datasets (or after train_data is fully built), add:
    print("=== CHECKING ALL SAMPLE TYPES ===")
    print("Total samples:", len(train_data))
    print("Column names:", train_data.column_names)

# Check a few samples across the dataset to find the spam source
    for idx in [0, 1, 2, 100, 200, 500]:
        if idx >= len(train_data):
            break
        s = train_data[idx]
        print(f"\n--- Sample {idx} ---")
        print("dataset_id:", s["dataset_id"])
        print("Safe_Model type:", type(s["Safe_Model"]))
        print("User type:", type(s["User"]))
        print("User value:", str(s["User"])[:80])
    # Try formatting it the same way prompt_formatting_func would
        try:
            system = s["System"][0] if isinstance(s["System"], list) else s["System"]
            user = s["User"][0][0] if isinstance(s["User"][0], list) else s["User"][0]
            print("system[:50]:", str(system)[:50])
            print("user[:50]:", str(user)[:50])
        except Exception as e:
            print("FORMAT ERROR:", e)

    if restricted_trainingset_size is not None:
        train_data = train_data.select(range(restricted_trainingset_size))

    return train_data, eval_data


def get_dataset_text_and_target_field():
    dataset_text_field = "User"
    dataset_target_field = "Model"
    return dataset_text_field, dataset_target_field
