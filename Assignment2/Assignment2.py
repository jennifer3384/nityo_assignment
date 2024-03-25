#!/usr/bin/env python
# coding: utf-8

# In[4]:


#pip install PyMuPDF pandas openpyxl
import fitz  # PyMuPDF
import pandas as pd


# In[5]:


#functions that extract text into excel file
def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Extract text from each page
    text_list = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        text_list.append(text)
    
    # Close the document
    doc.close()
    
    return text_list

def create_dataframe(text_list):
    # Create a DataFrame from the list of text
    df = pd.DataFrame({'Page Number': range(1, len(text_list) + 1), 'Text': text_list})
    return df


# In[8]:


#extract text for Chinses pdf
pdf_path_Chinese = 'C:/job hunting/nityo/Chinese_NLP.pdf'
Chinese_list = extract_text_from_pdf(pdf_path_Chinese)
Chinese_dataframe = create_dataframe(Chinese_list)

#extract text for English pdf
pdf_path_English = 'C:/job hunting/nityo/English_NLP.pdf'
English_list = extract_text_from_pdf(pdf_path_English)
English_dataframe = create_dataframe(English_list)


# In[16]:


#pip install googletrans==4.0.0-rc1
#pip install tqdm
from tqdm import tqdm
from googletrans import Translator, LANGUAGES


# In[17]:


def translate_text(text_list, src='zh-tw', dest='en'):
    translator = Translator()
    translated_texts = []
    
    # Wrap text_list in tqdm to show a progress bar
    for text in tqdm(text_list, desc="Translating"):
        # Check if text is not None or empty
        if text and not text.isspace():
            try:
                # Perform translation
                translated = translator.translate(text, src=src, dest=dest)
                translated_texts.append(translated.text)
            except Exception as e:
                # Handle translation error
                print(f"Error translating text: {e}")
                translated_texts.append("")  # Append empty string in case of error
        else:
            # Handle None or empty string
            translated_texts.append("")  # Append empty string for None or empty inputs
    return translated_texts


# In[19]:


# Translate text from Chinese to English
translated_texts = translate_text(Chinese_list)
translated_dataframe = create_dataframe(translated_texts)


# In[30]:


#merge the traslated dataset with original English dataset
merged_dataframe = pd.merge(translated_dataframe, English_dataframe, on='Page Number', suffixes=('_Translated', '_English'))
merged_dataframe


# In[32]:


#output the dataset as an excel file
excel_path='C:/job hunting/nityo/NLP_merged.xlsx'
merged_dataframe.to_excel(excel_path, index=False)


# In[ ]:




