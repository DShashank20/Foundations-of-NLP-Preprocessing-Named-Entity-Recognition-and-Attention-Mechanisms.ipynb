# Foundations of NLP: Preprocessing, Named Entity Recognition & Attention Mechanisms.   

## Name : SHASHANK REDDY DASARI   
## ID: 700781569       

## 📌 Project Overview

This project explores three fundamental concepts in **Natural Language Processing (NLP)** using Python. It demonstrates how text can be processed, how named entities can be extracted from sentences, and how attention mechanisms help models understand relationships between words.

The project is implemented using Python, NLTK, spaCy, and NumPy in a Jupyter Notebook.

## 🎯 Objectives

- Understand the NLP preprocessing pipeline.
- Perform tokenization, stopword removal, and stemming.
- Extract named entities using spaCy.
- Identify entity labels and character positions.
- Implement Scaled Dot-Product Attention from scratch.
- Understand how attention weights help capture contextual relationships.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NLTK | Text preprocessing |
| spaCy | Named Entity Recognition |
| NumPy | Numerical computations |
| Jupyter Notebook | Development environment |

## 📂 Project Structure

```text
NLP-Foundations/
│
├── Foundations_of_NLP_Preprocessing,
│   _Named_Entity_Recognition,
│   _and_Attention_Mechanisms.ipynb
│
└── README.md
```

## 🧠 Tasks Implemented

### Q1. NLP Preprocessing Pipeline

Implemented a basic NLP preprocessing pipeline using NLTK.

**Operations performed:**

1. **Tokenization:** Splits sentences into individual words and tokens.
2. **Stopword Removal:** Removes common English words such as "are", "in", and "and".
3. **Stemming:** Reduces words to their root forms using PorterStemmer.

**Example Input:**

```text
NLP techniques are used in virtual assistants like Alexa and Siri.
```

**Example Output:**

```text
Original Tokens:
['NLP', 'techniques', 'are', 'used', 'in', 'virtual',
 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']

Tokens Without Stopwords:
['NLP', 'techniques', 'used', 'virtual', 'assistants',
 'like', 'Alexa', 'Siri', '.']

Stemmed Words:
['nlp', 'techniqu', 'use', 'virtual', 'assist',
 'like', 'alexa', 'siri', '.']
```

### Q2. Named Entity Recognition (NER) with spaCy

Implemented Named Entity Recognition using the spaCy English language model.

NER identifies important real-world entities such as:
- PERSON – Names of people.
- ORG – Organizations.
- GPE – Countries, cities, and geographical locations.
- DATE – Dates and time expressions.

**Input Sentence:**

```text
LeBron James as the NBA player of the United States
in Laker and won the NBA Final MVP in 2020.
```

**Detected Named Entities:**

| Entity | Label | Start | End |
|---|---|---:|---:|
| LeBron James | PERSON | 1 | 13 |
| NBA | ORG | 21 | 24 |
| the United States | GPE | 35 | 52 |
| Laker | GPE | 56 | 61 |
| MVP | ORG | 84 | 87 |
| 2020 | DATE | 91 | 95 |

**Key Features:**
- Extracts named entities from text.
- Identifies entity categories.
- Displays start and end character positions.
- Uses spaCy's pretrained English NLP model.

### Q3. Scaled Dot-Product Attention

Implemented the Scaled Dot-Product Attention mechanism using NumPy.

The attention mechanism calculates relationships between Query (Q), Key (K), and Value (V) matrices.

**Formula:**

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

**Implementation Steps:**

1. Calculate the dot product between Q and transposed K.
2. Scale the scores by the square root of the key dimension.
3. Apply the Softmax function to calculate attention weights.
4. Multiply attention weights by the Value matrix.
5. Return the attention weights and final output.

**Test Results:**

Attention Weights Matrix:

```text
[[0.73105858 0.26894142]
 [0.26894142 0.73105858]]
```

Final Output Matrix:

```text
[[2.07576569 3.07576569 4.07576569 5.07576569]
 [3.92423431 4.92423431 5.92423431 6.92423431]]
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd NLP-Foundations
```

### 3. Install Required Libraries

```bash
pip install nltk spacy numpy jupyter
```

### 4. Download NLTK Resources

```python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 5. Download spaCy English Model

```bash
python -m spacy download en_core_web_sm
```

### 6. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook:

`Foundations_of_NLP_Preprocessing,_Named_Entity_Recognition,_and_Attention_Mechanisms.ipynb`

Run the cells sequentially to execute all three tasks.

⭐ If you find this project useful, feel free to star the repository!
