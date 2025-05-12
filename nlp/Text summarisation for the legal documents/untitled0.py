

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
import re


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_legal_text(text):
    text = re.sub(r'\n+', ' ', text)            
    text = re.sub(r'\s+', ' ', text).strip()    
    return text


def summarize_legal_text(text, num_sentences=3):
    cleaned_text = clean_legal_text(text)
    parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join(str(sentence) for sentence in summary)

legal_document = """
THIS AGREEMENT is made and entered into as of the Effective Date by and between Party A and Party B.
WHEREAS, Party A is engaged in the business of providing certain services, and Party B desires to engage Party A to perform such services.
NOW, THEREFORE, in consideration of the mutual covenants and promises herein contained, the parties agree as follows:
1. Services: Party A agrees to perform the services described in Exhibit A.
2. Payment: Party B shall compensate Party A as set forth in Exhibit B.
3. Term: This Agreement shall commence on the Effective Date and continue for a period of one year.
4. Termination: Either party may terminate this Agreement with thirty (30) days written notice.
5. Governing Law: This Agreement shall be governed by and construed in accordance with the laws of the State of Example.
"""


summary = summarize_legal_text(legal_document, num_sentences=3)


print("🔍 Legal Document Summary:\n")
print(summary)
