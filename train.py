from trl import SFTTrainer, SFTConfig
from data import load_data, load_model

## Training the LoRA params (matrices)

## each dataset dictionary is in form replacing input 
# with either output or instruction as well ['train'][index]['input']
model = load_model()
training_args = SFTConfig(output_dir="checkpoints", dataset_text_field="input", learning_rate=2e-4,     warmup_steps=100,
    num_train_epochs=3,
    logging_steps=10,
    weight_decay=0.01)
trainer = SFTTrainer(
    model=model,
    train_dataset = load_data(),
    args = training_args
)
trainer.train()
trainer.save_model("model")