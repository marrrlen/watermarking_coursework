from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto")

inputs = "Mathematical linguistics is"
input_tokenized = tokenizer(inputs, return_tensors="pt").to(model.device)

outputs = model.generate(**input_tokenized, max_new_tokens=256, num_beams=8, no_repeat_ngram_size=4) 
generated_text = tokenizer.decode(outputs[0][input_tokenized["input_ids"].shape[-1]:], skip_special_tokens=True)
print(generated_text)
