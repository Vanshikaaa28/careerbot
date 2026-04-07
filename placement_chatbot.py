import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model
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

# Customize bot identity
bot_name = "CareerBot"

print("CareerBot: Hello! I can help you with placements, resumes, and interviews.")
print("Type 'quit' to exit.\n")

exit_commands = ["quit", "exit", "bye"]

fallback_responses = [
    "I'm not sure I understood that. Could you rephrase?",
    "Sorry, I don't have information on that yet.",
    "Please try asking about placements, resumes, or interviews."
]

while True:

    sentence = input("You: ")

    if sentence.lower() in exit_commands:
        print("CareerBot: Goodbye! Best of luck with your career.")
        break

    # Process input
    sentence = tokenize(sentence)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Slightly relaxed threshold
    if prob.item() > 0.70:

        for intent in intents['intents']:

            if tag == intent["tag"]:

                response = random.choice(intent['responses'])

                print(f"{bot_name}: {response}")

    else:

        print(f"{bot_name}: {random.choice(fallback_responses)}")