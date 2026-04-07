# CareerBot: AI-Powered Placement Guidance Chatbot using PyTorch

CareerBot is a contextual chatbot implementation using PyTorch designed to assist students with placement preparation, interview guidance, resume tips, and general career-related queries.

* The implementation is simple and easy to understand for beginners.
* The chatbot uses a Feed Forward Neural Network with hidden layers for intent classification.
* The dataset is customizable for different career and placement-related use cases.
* Users can modify `intents.json` to extend the chatbot’s knowledge and retrain the model.

This project demonstrates the practical use of Natural Language Processing (NLP), deep learning, and conversational AI using Python and PyTorch.

---

## Features

* Intent classification using a neural network
* NLP preprocessing using tokenization and stemming
* Bag-of-Words text vectorization
* Custom placement and interview guidance dataset
* Confidence-based response system
* Easy dataset customization
* Beginner-friendly implementation

---

## Project Structure

careerbot-placement-chatbot/

train.py — Script to train the neural network
placement_chatbot.py — Main chatbot application
model.py — Neural network model definition
nltk_utils.py — Text preprocessing utilities
intents.json — Dataset containing patterns and responses
data.pth — Trained model file
README.md — Project documentation

---

## Installation

### Create a Virtual Environment

You can use `venv` or `conda`.

```console
mkdir careerbot
cd careerbot
python -m venv venv
```

### Activate the Environment

Windows:

```console
venv\Scripts\activate
```

Mac / Linux:

```console
. venv/bin/activate
```

---

## Install Dependencies

Install required libraries:

```console
pip install torch
pip install nltk
pip install numpy
```

If you encounter an error during the first run, install the tokenizer:

```console
python
>>> import nltk
>>> nltk.download('punkt')
```

---

## Usage

### Step 1: Train the Model

Run:

```console
python train.py
```

This will generate:

```console
data.pth
```

---

### Step 2: Run the Chatbot

```console
python placement_chatbot.py
```

Example interaction:

You: What companies visit campus?
CareerBot: Top companies visiting campus include TCS, Infosys, and Accenture.

You: How do I prepare for interviews?
CareerBot: Practice coding problems, review core subjects, and prepare HR questions.

You: quit
CareerBot: Goodbye! Best of luck with your career.

---

## Customize the Chatbot

You can modify the chatbot behavior by editing:

```console
intents.json
```

Add:

* New tags
* User input patterns
* Bot responses

Example:

```console
{
  "tag": "interview_preparation",
  "patterns": [
    "How should I prepare for interviews?",
    "Interview tips",
    "How to crack interviews"
  ],
  "responses": [
    "Practice coding regularly and revise core subjects.",
    "Prepare common HR questions and mock interviews."
  ]
}
```

After modifying the dataset, retrain the model:

```console
python train.py
```

---

## Technologies Used

Python
PyTorch
Natural Language Processing (NLP)
NLTK
NumPy
JSON

---

## Skills Demonstrated

Machine Learning
Deep Learning
Natural Language Processing
PyTorch Model Development
Text Preprocessing
Chatbot Development
Python Programming

---

## Future Improvements

* Add web interface using Flask or Streamlit
* Deploy chatbot to cloud platform
* Add voice input support
* Improve model accuracy with larger dataset
* Use advanced NLP models (e.g., transformers)

---

## Author

**Vanshika Jaiswal**
3rd Year B.Tech Student — CSE (AI & ML)

* 📄 Resume: [https://drive.google.com/file/d/1tAGPmbdzw5Czp2w49jSamslzd_JHOFpi/view](https://drive.google.com/file/d/19eXdpldPJjVSddwX_yz6HiXOZJb3Fk1H/view?usp=sharing)
* 🔗 LinkedIn: https://www.linkedin.com/in/vanshikaaa/
* 💻 GitHub: https://github.com/Vanshikaaa28

---

## Acknowledgement

This project is inspired by an open-source PyTorch chatbot implementation and has been customized with a new dataset and functionality for placement guidance and career support.

---

## License

This project is intended for educational and demonstration purposes.
