from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class SentencePolishingGenertor:
    def __init__(self):
        model_name = "kssumanth6/t5_small_sentence_polishing_generator_v3"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate_response(self, input_text, context, user_profile):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response