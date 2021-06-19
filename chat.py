import random
import json
import pandas as pd
import torch
import time
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intent.json', 'r') as json_data:
    intents = json.load(json_data)
df = pd.read_csv('dataset/JEP.csv')

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

amount = 0
bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

def ask(sunn):
    question = df[' Question'][sunn]
    hint = df[' Category'][sunn]
    answer = df[' Answer'][sunn]
    print('Next question. Hint: {}. \n {} \n Answer is {}?'.format(hint, question, answer))
    #print("10s job current time : {}".format(time.ctime()))
    time.sleep(10)
    
while True:
    # sentence = "do you use credit cards?"
    word = input("You: ")
    if word == "quit":
        break

    sentence = tokenize(word)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                if tag == 'skip':
                    sunn = random.randint(0, 43387)
                    ask(sunn)

    else:
        #print(f"{bot_name}: I do not understand...")
        if word in df[' Answer'][sunn]:
            ignore = ['$', ',']
            amount += int(''.join([i for i in df[' Value'][sunn] if i not in ignore]))
            print(df[' Round'][sunn], 'The Answer Is ', df[' Answer'][sunn])
            print('Correct, you got {} with a total of ${}'.format(df[' Value'][sunn], amount))
            #print("10s job current time : {}".format(time.ctime()))
            time.sleep(10)
            sunn = random.randint(0, 43387)
            ask(sunn)
        else: 
            print('wrong answer. Answer is {}'.format(df[' Answer'][sunn]))
            sunn = random.randint(0, 43387)
            ask(sunn)
        
