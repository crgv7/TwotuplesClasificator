import pandas as pd
from pysentimiento import create_analyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import asent

def PysentimentClasificator(data:str, ColumnName:str):
  df = pd.read_excel(data, sheet_name='Sheet1')
  resultado={}
  analyzer = create_analyzer(task="sentiment", lang="es")
  for index, row in df.iterrows():
    out=analyzer.predict(row[ColumnName])#clasifica el texto
    resultado[index]=out.output
    print(index)


  frame=pd.DataFrame( {'clasificacion_pysentimiento': resultado})   #crear nuevo dataframe con una columna de clasificacion
  frame.to_excel('score_pysentiment.xlsx')

def BertClasificator(data:str, ColumnName:str):
  df = pd.read_excel(data, sheet_name='Sheet1')
  bert_clasif={}
  
  # Cargamos el modelo y el tokenizer
  model_id = "nlptown/bert-base-multilingual-uncased-sentiment"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForSequenceClassification.from_pretrained(model_id)
  
  for index, row in df.iterrows():
    opinion = str(row[ColumnName])
    
    # Tokenización
    inputs = tokenizer(opinion, return_tensors="pt", padding=True, truncation=True)
    
    # Inferencia
    with torch.no_grad():
      outputs = model(**inputs)
      
    # Procesar resultados
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    estrellas = prediction + 1
    
    print(f"{index} --- Estrellas: {estrellas}")
    if estrellas in [1, 2]:
      bert_clasif[index] = 'NEG'
    elif estrellas == 3:
      bert_clasif[index] = 'NEU'
    elif estrellas in [4, 5]:
      bert_clasif[index] = 'POS'

  frame=pd.DataFrame( {'Clasificacion_Bert': bert_clasif})
  frame.to_excel('score_bert.xlsx')

def AsentClasificator(data:str, ColumnName:str, C=True):
  # Como se quitó la traducción, leemos directamente el archivo original
  df = pd.read_excel(data, sheet_name='Sheet1')

  asentimiento_clasif={}
  # load spacy pipeline
  nlp = spacy.blank('en')
  nlp.add_pipe('sentencizer')

  # add the rule-based sentiment model
  nlp.add_pipe('asent_en_v1')


  for index, row in df.iterrows():
      opinion = str(row[ColumnName])
      sentiment=nlp(opinion)#clasifica el texto
      asentimiento_clasif[index]=sentiment._.polarity
      print(asentimiento_clasif[index])
      print(index)
      if asentimiento_clasif[index].compound >0.2:
        asentimiento_clasif[index]='POS'
      elif asentimiento_clasif[index].compound<0.2 and asentimiento_clasif[index].compound>0 :
        asentimiento_clasif[index]='NEU'
      else:
        asentimiento_clasif[index]='NEG'

  frame=pd.DataFrame( {'Clasificacion_Asentiment': asentimiento_clasif})   #crear nuevo dataframe con una columna de clasificacion
  frame.to_excel('score_asent.xlsx')
