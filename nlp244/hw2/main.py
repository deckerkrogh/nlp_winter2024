from argparse import ArgumentParser
import torch
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer, \
    AutoTokenizer, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, \
    AutoModelForSequenceClassification, pipeline, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import evaluate
from sklearn.metrics import classification_report
import math
import pandas as pd

# Set to GPU if available.
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using CPU instead.')


def train_model(data_path, save_model_path):
    wiki_dataset = load_dataset("wikitext", 'wikitext-2-raw-v1', download_mode="force_redownload", ignore_verifications=True)

    tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    def preproc_data(data):
        return tokenizer([s for s in data["text"]])

    tokenized_wiki_dataset = wiki_dataset.map(preproc_data, batched=True, remove_columns=wiki_dataset["train"].column_names)

    block_size = 128

    def group_texts(examples):
        # This partially copied from hugging face causal modeling tutorial.
        # Concatenate all the tokens in a batch
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Cut off any extra that won't be divisible by block size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # result["labels"] = result["input_ids"].copy()
        return result

    grouped_dataset = tokenized_wiki_dataset.map(group_texts, batched=True)

    # Create model
    data_collator_lm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    N_HEAD = 4
    N_DIM = 128
    N_LAYER = 4
    VOCAB_SIZE = tokenizer.vocab_size
    configuration = OpenAIGPTConfig(n_head=N_HEAD, n_embd=N_DIM, n_layer=N_LAYER, vocab_size=VOCAB_SIZE)
    pre_model = OpenAIGPTLMHeadModel(configuration)

    training_args = TrainingArguments(
        output_dir=save_model_path,
        evaluation_strategy='epoch',
        disable_tqdm=False,
        learning_rate=1e-4,
        num_train_epochs=1,
        log_level='info',
        logging_strategy='epoch',
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64)

    pre_trainer = Trainer(
        model=pre_model,
        args=training_args,
        train_dataset=grouped_dataset['train'],
        eval_dataset=grouped_dataset['validation'],
        # compute_metrics=compute_metrics,
        data_collator=data_collator_lm
    )

    print("pre-training...")
    pre_trainer.train()
    print("finished pretraining\n\n")

    eval_results = pre_trainer.evaluate()
    print(f"Pretraining perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # model_to_save = pre_model.module if hasattr(pre_model, 'module') else pre_model
    pre_trainer.save_model(save_model_path)

    # ---------------- Finetuning ------------------
    intent_data = load_dataset("csv", data_files=data_path)['train']
    intent_data = intent_data.train_test_split(test_size=0.2)
    intent_train = intent_data['train'].flatten()
    intent_val = intent_data['test'].flatten()

    all_relations = list(
        sorted(set([j for d in intent_train["Core Relations"] if d is not None for j in d.split()] + ["None"])))
    relation_to_idx = {r: i for i, r in enumerate(all_relations)}
    idx_to_relation = {i: r for i, r in enumerate(all_relations)}

    def preproc_intent_data(d):
        # Tokenize
        res = tokenizer(d['utterances'], max_length=50, padding='longest', truncation=True)

        # Encode core relations
        if d["Core Relations"] is None:
            res['label'] = relation_to_idx["None"]
        else:
            for r in d["Core Relations"].split():
                res["label"] = relation_to_idx[r]
        return res

    # intent_data = intent_data.map(preproc_intent_data, remove_columns=intent_train.column_names)
    intent_train_tok = intent_train.map(preproc_intent_data, remove_columns=intent_train.column_names)
    intent_val_tok = intent_val.map(preproc_intent_data, remove_columns=intent_val.column_names)

    ft_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ft_model = AutoModelForSequenceClassification.from_pretrained(save_model_path, num_labels=len(all_relations),
                                                                  id2label=idx_to_relation, label2id=relation_to_idx)
    ft_model.resize_token_embeddings(len(tokenizer))
    ft_model.config.pad_token_id = tokenizer.pad_token_id
    ft_model.to(device)
    ft_training_args = TrainingArguments(
        output_dir='ft_trained_hw2_model',
        evaluation_strategy='epoch',
        disable_tqdm=False,
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        log_level='info'
    )

    ft_trainer = Trainer(
        model=ft_model,
        args=ft_training_args,
        train_dataset=intent_train_tok,
        eval_dataset=intent_val_tok,
        data_collator=ft_data_collator,
        # compute_metrics=compute_metrics
    )

    print("finetuning...")
    ft_trainer.train()
    print("done finetuning\n")
    ft_trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)


def test_model(data_path, model_path, output_path):
    intent_data = load_dataset("csv", data_files=data_path)['train']
    # Evaluate
    print(intent_data)
    infer = pipeline(task="text-classification", model=model_path)
    val_preds = [infer(utt)[0]["label"] for utt in intent_data["utterances"]]
    # val_true = [relation for relation in intent_data["Core Relations"]]
    # val_true = ['None' if relation is None else relation for relation in val_true]
    # print(classification_report(val_preds, val_true, zero_division=0.0))

    # save the predictions as a CSV
    df = pd.read_csv(data_path)
    df['predictions'] = val_preds
    df.to_csv(output_path)


if __name__ == "__main__":
    parser = ArgumentParser("homework CLI")

    parser.add_argument('--train', action="store_true", help="indicator to train model")
    parser.add_argument('--test', action="store_true", help="indicator to test model")

    parser.add_argument('--data', help="path to data file")
    parser.add_argument('--save_model', help="ouput path of trained model")
    parser.add_argument('--model_path', help="path to load trained model from")

    parser.add_argument('--output', help="output path of predictions")

    args = parser.parse_args()

    if args.train:
        train_model(args.data, args.save_model)
    if args.test:
        test_model(args.data, args.model_path, args.output)
