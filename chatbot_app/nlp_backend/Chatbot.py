from transformers import AutoModelForTokenClassification, AutoTokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import scipy.spatial
from sentence_transformers import SentenceTransformer


class Chatbot:
    def __init__(self, ner_model_name, intent_model_name, generators):
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
        self.intent_model = AutoModelForTokenClassification.from_pretrained(intent_model_name)
        self.generators = generators
        self.context = []
        self.user_profile = {}
        self.max_context_length = 10
        self.id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}
        self.label2intent = {0: "empathetic", 1: "facts", 2: "chitchat"}
        self.embedder = SentenceTransformer("bert-base-nli-mean-tokens")

    def update_user_profile(self, entities):
        for entity in entities:
            if entity[1] == 'PER':
                self.user_profile['name'] = entity[0]

    def set_context(self, context):
        self.context = context

    def classify_intent(self, message):
        inputs = self.intent_tokenizer(message, return_tensors="pt")
        outputs = self.intent_model(**inputs)
        logits = outputs.logits
        averaged_logits = logits.mean(dim=1)
        probabilities = torch.softmax(averaged_logits, dim=-1)
        intent_id = torch.argmax(probabilities, dim=-1).item()

        intent = self.label2intent[intent_id]
        return intent

    def rerank_responses(self, responses, query):
        if not responses:
            return responses
        corpus_embeddings = self.embedder.encode(responses)
        query_embedding = self.embedder.encode([query])
        distances = scipy.spatial.distance.cdist(query_embedding, corpus_embeddings, "cosine")[0]
        sorted_indices = np.argsort(distances)
        sorted_responses = [responses[idx] for idx in sorted_indices]
        return sorted_responses[-1]

    def extract_entities(self, message):
        inputs = self.ner_tokenizer(message, return_tensors="pt")
        outputs = self.ner_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        tokens = inputs["input_ids"].squeeze().tolist()
        entities = []
        current_entity = ""
        current_label = ""
        for token, prediction in zip(tokens, predictions.squeeze().tolist()):
            label = self.id2label[prediction]
            token_str = self.ner_tokenizer.convert_ids_to_tokens(token)
            if label == 'O' or token_str in ['[SEP]', '[CLS]']:
                if current_entity:
                    entities.append((current_entity, current_label))
                    current_entity = ""
                    current_label = ""
                continue
            if token_str.startswith("##"):
                token_str = token_str[2:]
            else:
                if current_entity:
                    entities.append((current_entity, current_label))
                    current_entity = ""
                    current_label = ""
            if label.startswith("B-") or not current_entity:
                current_entity = token_str
                current_label = label[2:]
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity += token_str
        if current_entity:
            entities.append((current_entity, current_label))

        return entities

    def retrieve_context(self):
        if self.context:
            context_length = 5
            return self.context[-context_length:]
        else:
            return []

    def generate_response(self, user_message):
        if 'what is my name' in user_message.lower():
            if 'name' in self.user_profile:
                response = f"Your name is {self.user_profile['name']}."
            else:
                response = "I'm not sure what your name is. What should I call you?"
            return response
        intent = self.classify_intent(user_message)
        context = self.retrieve_context()
        print("context : ", context)
        responses = {}

        if intent == "empathetic":
            responses['empathetic'] = self.generators['empathetic'].generate_response(user_message, context,
                                                                                      self.user_profile)
        else:
            responses['chitchat'] = self.generators['chitchat'].generate_response(user_message, context,
                                                                                  self.user_profile)

        responses['factual'] = self.generators['facts'].generate_response(user_message, context, self.user_profile)
        best_response = self.rerank_responses([responses.get(key) for key in responses if responses.get(key)],
                                              user_message)
        best_response = self.generators['sentencepolishing'].generate_response(best_response, context,
                                                                               self.user_profile)
        return best_response

    def clean_response(self, response):
        response = response.replace('\\n', '\n')
        response = response.replace('"', '')
        response = response.replace('\\n', ' ')
        return response