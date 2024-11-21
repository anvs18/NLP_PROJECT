# NLP_PROJECT: Mental Health Chatbot

# Introduction:  
This project aims to develop an end-to-end conversational system, termed a socialbot, with the capability to engage in open-domain chitchat and human-like conversations with empathy. The system is designed to receive user input in the form of a query text and generate text responses that are contextually appropriate to the query and ongoing conversation. This project explores the utilization of Large Language Models (LLMs) in the development of a Mental Health Support Chatbot that addresses the mental health crisis by developing an accessible conversational system using Natural Language Processing techniques. The goal is
to develop a chatbot that can quickly and meaningfully understand and respond to a variety of emotions and mental health concerns in order to address issues such as people’s embarrassment about discussing mental health concerns with others and a lack of resources.  

Detailed Project Report and Analysis: [Paper](https://github.com/alla-sahithya/Mental-Health-Chatbot/blob/main/report.pdf)

# Datasets used:  
1. BYU PCCL chitchat the Amazon Alexa Topical Chat dataset
2. Empathetic Dialogues  
3. Chatbot Mental Health conversations
4. General-Knowledge dataset from hugging face
5. CoNLL-2003 dataset
6. PAWS (Paraphrase Adversaries from Word Scrambling)
   
The dataset required to run the notebooks is present in the "[data](https://github.com/alla-sahithya/Mental-Health-Chatbot/tree/main/data)" folder.

# Main Components:  
1. Empathetic Generator  
2. Chit-chat Generator
3. Facts Generator
4. Neural re-ranker
5. Dialouge Manager
6. Intent Classifier
7. NER module  

# System Architecture:  
![image](https://github.com/user-attachments/assets/8cc60248-cf1d-42ff-94fb-6511b1911ff1)  

The preprocessing, model training, and evaluation code are present within the same notebook for NER, Intent Classification, Empathetic Generator, and facts generator.
However, the Chitchat and sentence polishing modules are housed in separate files for their model training and evaluation.

Contributors:
Naga Venkata Sahithya Alla - anvs18, sahithya-alla and
Surya Sumanth Karuturi - suryanit


P.S: The code and folders structure is based on the instructions given by our instructor in the course. 
