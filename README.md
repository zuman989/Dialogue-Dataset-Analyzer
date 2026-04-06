## Dialogue Dataset Analyzer

A Python-based NLP tool for analyzing dialogue datasets and extracting persona traits and conversational patterns.



## Features
- Sentiment analysis using VADER  
- Speaker behavior analysis  
- Persona trait extraction from dialogue  
- Turn-taking and interaction pattern analysis  
- Generates structured reports in HTML and JSON formats  



## Tech Stack
- Python  
- spaCy  
- NLTK (including VADER)  
- Pandas  
- Matplotlib  
- HTML  



## What it does

This project processes dialogue datasets to identify patterns in how people communicate. It looks at sentiment, language use, and interaction structure to infer behavioral traits and persona characteristics.

It can be used for:
- analyzing conversational data  
- understanding dialogue behavior  
- supporting chatbot or conversational AI development  



## How to run

Clone the repository and install dependencies:

git clone https://github.com/zuman989/Dialogue-Dataset-Analyzer.git  
cd Dialogue-Dataset-Analyzer  
pip install -r requirements.txt  

Run the analysis: python main.py  

Generate reports: python report.py  



## Project structure

data/       input datasets  
reports/    generated outputs  
src/        core logic  
main.py     runs analysis  
report.py   generates reports  



## Future work
- Improve visualization and reporting  
- Add support for larger datasets  
- Extend persona modeling capabilities  
- Explore integration with LLM-based systems  



## Author
Zuman
