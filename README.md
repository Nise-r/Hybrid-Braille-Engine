# Hybrid Braille Translation System

## 📌Overview

This project implements the core logic of a hybrid architecture for translating text from Indian languages into Bharti Braille.

The architecture is based on the research paper:

Nisheeth Joshi & Pragya Katyayan, "A Model for Translation of Text from Indian Languages to Bharti Braille Characters".

The system integrates rule-based transliteration with a deep learning disambiguation model (Bi-directional LSTM) to achieve near-100% accuracy across multiple Indian languages and scripts.

## 🎨 Demo



https://github.com/user-attachments/assets/e004c940-9a58-4afe-ad2b-6bb3aabb53e7





## ⚙️System Architecture:
The hybrid model consists of two main components:

-  Rule-Based Transliteration(Liblouis):
    - Extracts phonemes from the input text.
    - Maps each character/phoneme to its Bharti Braille encoding.
    - Uses predefined mappings for:
        - Consonants
        - Vowels
        - Vowel symbols (Matra/Diacritics)
    - Handles the majority of conversions directly.
-  Bi-directional LSTM Disambiguator:
    - Triggered only when multiple Braille encodings exist for a character (ambiguity cases).
    - Uses context from surrounding characters to choose the correct encoding.
    - Architecture:
        - Embedding layer for token representation.
        - Forward & backward LSTM layers to capture context in both directions.
        - Dropout layer to reduce overfitting.
        - Fully connected layer for prediction.
      
## 🔄Workflow:

- Input Text (any supported Indian language)
- Phoneme Extraction → Convert input text into phoneme sequence.
- Rule-Based Mapping → Map phonemes to Bharti Braille cells.
- Ambiguity Detection → If a word has ambiguious part that has >1 possible encoding:
    - Send the token sequence to the Bi-LSTM model.
- Contextual Disambiguation → Model predicts the correct Braille symbol.
- Final Braille Output.

## 📊Evaluation:
- Dataset: 2900 words(Currently Hindi only).
- Accuracy:
    - Rule-based system: ~92–94% (varies by script)
    - LSTM: 100% accuracy in resolving ambiguities
    - Overall system accuracy: 100%
    
## 📂Project Structure:
```bash
├── data/                    # Language data 
├── model/                   # Saved LSTM models
├── braille_files/
│   ├── bharati_braille.cti
│   ├── braille-patterns.cti
│   ├── braille_patterns.txt
├── notebooks/               # Training and testing experiments
├── README.md                # Documentation
├── requirements.txt         # Dependencies
├── main.py                  # File to setup fastapi server
├── gui_app.py               # GUI app
├── install_liblouis.ps1     # Installation script for Windows
└── install_liblouis.sh      # Installation script for Linux
```
## 🚀Installation & Setup:
```bash
#Clone repository
git clone https://github.com/flickdone/ai-rajat-sharma.git
cd ai-rajat-sharma

# Install dependencies
pip install -r requirements.txt

#For windows run:
.\install_liblouis.ps1

#For Linux run:
./install_liblouis.sh

```

## ▶️Usage Example:
```bash
#For GUI application:
python gui_app.py

#For fastapi server:
fastapi dev main.py

#After setting up fastapi server you can send request to server:
curl -X POST http://127.0.0.1:8000/api/v1/braille -H "Content-Type: application/json" -d '{"text":"HINDI_TEXT"}'
```

## 📈Future Improvements:
- Expand support for more languages(currently only works on Hindi).
