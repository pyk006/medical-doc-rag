from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraModel, LoraConfig, get_peft_model



def load_data():
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    return dataset
def load_rag_data():
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    return dataset
def load_model():
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-chat-v1.0", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-chat-v1.0")

# - - - LoRA - - - #
# These are the actual learnable matrices
# while everything else is frozen

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01
    )

    model = get_peft_model(model, config)
    return model