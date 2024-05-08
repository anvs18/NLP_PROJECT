from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class EmpatheticGenerator:
    def __init__(self):
        model_name = "nlpproject/t5small_EmpatheticChatbot_ED3"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate_response(self, input_text, context, user_profile):
        context_texts = [turn['user'] + ' ' + turn['bot'] for turn in context]
        input_sequence = " ".join(context_texts + [input_text])
        print("input sequence: " + input_sequence)
        input_ids = self.tokenizer.encode(input_sequence, return_tensors='pt')
        outputs = self.model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response