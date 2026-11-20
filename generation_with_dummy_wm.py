#!git clone https://github.com/jwkirchenbauer/lm-watermarking.git
import sys
sys.path.append("/content/lm-watermarking")
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from extended_watermark_processor import WatermarkLogitsProcessor

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto")

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=0.0,
                                               seeding_scheme="minhash") #delta (константа, прибавляемая к логитам из зеленого списка) взята за 0,
                                               #поэтому вотермарка не будет накладываться
                                               #значения остальных параметров взяты исходя из рекомендаций авторов 
                                               #(из-за selfhash генерация занимает больше времени,
                                               #поэтому пока что можно заменить на minhash (7 минут vs 30 секунд))

inputs = "Mathematical linguistics is"
input_tokenized = tokenizer(inputs, return_tensors="pt").to(model.device)

outputs = model.generate(**input_tokenized,
                               max_new_tokens=256, num_beams=8, no_repeat_ngram_size=4,
                               logits_processor=LogitsProcessorList([watermark_processor]))

outputs = outputs[:,input_tokenized["input_ids"].shape[-1]:]
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(generated_text)
