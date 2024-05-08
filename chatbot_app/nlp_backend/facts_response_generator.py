import torch
from transformers import AutoTokenizer, AutoModel

class FactsGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", embeddings_file='facts_embeddings_db.pt'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.embeddings_db = torch.load(embeddings_file)


    def encode_text(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings[0]

    def semantic_similarity(self, embedding1, embedding2):
        embedding1 = embedding1 / embedding1.norm()
        embedding2 = embedding2 / embedding2.norm()
        cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        return cosine_sim.item()

    def generate_response(self, user_query, context, user_profile):
        user_query_embedding = self.encode_text(user_query)
        highest_similarity = 0.4
        best_response = "I'm sorry, I don't have information on that topic."

        for fact in self.embeddings_db:
            fact_question_embedding = fact['embedding']
            similarity = self.semantic_similarity(user_query_embedding, fact_question_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_response = fact['context']
        return best_response