import random
import json
import torch
import time
import pandas as pd
from model import NeuralNet
from fastapi import FastAPI
from nltk_utils import bag_of_words, tokenize, stem

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('dataset/JEP.csv')
with open('intent.json', 'r') as json_data:
    intents = json.load(json_data)
    
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

model_state = data["model_state"]
    
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
all_words = []
tags = []

for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
# stem and lower each word
ignore_words = ['?', '.', '!', '\n', '*']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

def prediction(word):
    """This function is to make prediction from the model based on the sentence the user entered"""
    # preprocess the text
    sentence = tokenize(word)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # pass it to the model and get the prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    
    # get the tag and probability
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
        
    # return the tag and the probablity
    return prob, tag
    
# declaring the home page for the API
@app.get("/")
@app.get("/chat")
def ping():
    return {"message": "Hey, I am Game bot, Get ready to answer questions!"}

def ask(code):
    """This function is to generate the questions and hint based on a code which is just a random value representing the location of a particular question."""
    # get the corresponding question and hint based on the index (code)
    question = df[' Question'][code]
    hint = df[' Category'][code]
    answer = df[' Answer'][code]
    
    return question, hint, answer

def crtime(code):
    
    """ This function is to generate the correct answr, the value of money associated with it, the question type i.e Jeopardy or double jeopardy."""
    # total amount will be calculated in the app.
    Round = df[' Round'][code]
    ans = df[' Answer'][code]
    value = df[' Value'][code]
    
    return Round, ans, value
    
# gateway for prediction
@app.get("/predict/{text}")
def predict(text: str):
    """This i"""
    while True:
        if '_'in text:
            container = text.split('_')
            sentence = container[0]
            code = int(container[-1])
        else:
            sentence = text
                   
        prob, tag = prediction(sentence)
        
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    if tag == 'skip' or 'next':
                        code = random.randint(0, 43387)
                        question, hint, answer = ask(code)
                        
                        return {'response': response, 'question': 'Question. Hint: {}. \n {} \n Answer is {}'.format(hint, question, answer), 'Code: ':code}
                  
        else:
            if sentence in df[' Answer'][code]:
                Round, ans, value= crtime(code)
                return {"round":Round, "answer":str('The Answer Is '+ans), "value":str('Correct, you got {}'.format(value))}
            
            else: 
                
                return {'response': 'Wrong answer.', 'answer': 'Answer is {}. To continue, type start.'.format(df[' Answer'][code])}
