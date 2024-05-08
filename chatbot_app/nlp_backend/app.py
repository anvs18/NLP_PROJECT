from transformers import AutoModelForTokenClassification, AutoTokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import scipy.spatial
from sentence_transformers import SentenceTransformer
from flask import Flask, request
from flask_cors import CORS

from Chatbot import Chatbot
from chitchat_response_generator import ChitchatGenerator
from ed_response_generator import EmpatheticGenerator
from facts_response_generator import FactsGenerator
from sentence_polishing_response_generator import SentencePolishingGenertor

app = Flask(__name__)
CORS(app)

app.debug = True

@app.route("/chatbot", methods=["POST"])
def chatbot():
    generators = {
        "empathetic": EmpatheticGenerator(),
        "facts": FactsGenerator(embeddings_file='facts_embeddings_db.pt'),
        "chitchat": ChitchatGenerator(),
        "sentencepolishing": SentencePolishingGenertor()
    }
    chatbot = Chatbot("nlpproject/NER_distilBERT", "nlpproject/IntentClassification_V1", generators)

    data = request.get_json()
    user_input = "user: " + data.get("user_input")
    # current_context = data.get("context")
    # chatbot.set_context(current_context)
    chatbot_response = None

    if user_input.lower() in ["quit", "bye", "see you later"]:
        chatbot_response = "Goodbye!"
    elif user_input.lower() in ["thanks", "thank you", "thankyou"]:
        chatbot_response = "You are welcome!"

    name_intro_phrases = ["my name is ", "you can call me "]
    name = None
    for phrase in name_intro_phrases:
        if phrase in user_input.lower():
            start_index = user_input.lower().index(phrase) + len(phrase)
            name = user_input[start_index:].strip()
            break

    if name:
        response = f"Nice to meet you, {name}!"
        chatbot.user_profile['name'] = name

    if chatbot_response is None:
        chatbot_response = chatbot.clean_response(chatbot.generate_response(user_input))

    return {
        "chatbot_response": chatbot_response
    }

if __name__ == "__main__":
    app.run(port=8080)