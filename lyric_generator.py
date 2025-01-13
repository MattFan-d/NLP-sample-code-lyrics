import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Step 1: Initialize a fresh GPT-2 model and tokenizer
def initialize_model_and_tokenizer(vocab_size=50257):
    config = GPT2Config(
        vocab_size=vocab_size,  # Vocabulary size of the tokenizer
        n_embd=768,            # Embedding size
        n_layer=12,            # Number of transformer layers
        n_head=12,             # Number of attention heads
    )
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Step 2: Prepare the dataset
def prepare_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # MLM (Masked Language Modeling) is not used here
    )
    return dataset, data_collator

# Step 3: Train the model
def train_model(model, tokenizer, dataset, data_collator, output_dir, epochs=3, batch_size=4):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()

# Step 4: Generate new lyrics
def generate_lyrics(model, tokenizer, prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Main Execution
if __name__ == "__main__":
    # Set paths
    lyrics_file = "/Users/mfan/Documents/Collections/Notes/2018/Corpus Linguistics/billboard_lyrics_1964-2015- cleaned.csv"  # Path to your dataset file
    output_dir = "./custom_lyric_model"
    
    # Step 1: Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer()

    # Step 2: Prepare the dataset
    dataset, data_collator = prepare_dataset(lyrics_file, tokenizer)

    # Step 3: Train the model
    train_model(model, tokenizer, dataset, data_collator, output_dir, epochs=3)

    # Step 4: Generate new lyrics
    prompt = "Under the stars and moonlit skies"
    lyrics = generate_lyrics(model, tokenizer, prompt, max_length=100)
    for i, lyric in enumerate(lyrics):
        print(f"Lyrics {i+1}:\n{lyric}")
