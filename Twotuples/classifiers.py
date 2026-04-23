import pandas as pd
from pysentimiento import create_analyzer
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

def TraduccionText(data:str, ColumnName:str):
  en_text={}
  df = pd.read_excel(data, sheet_name='Sheet1')
  #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  translator = pipeline("translation_es_to_en",
                      model="Helsinki-NLP/opus-mt-es-en") #modelo
  for index, row in df.iterrows():
    try:
      if len(row[ColumnName])>390:
        opinion=row[ColumnName][0:390]
        traduccion=translator(opinion)
      else:
        traduccion=translator(row[ColumnName])
      en_text[index]=traduccion[0]['translation_text']
      print(index)
    except:
      print("no se pudo ejecutar")

  frame=pd.DataFrame( {'EN_Opiniones': en_text})
  frame.to_excel('score_en.xlsx')
  return frame

def VaderClasificator(data:str, ColumnName:str):
  df=TraduccionText(data,ColumnName)
  nltk.download('vader_lexicon')
  vader_clasif={}
  #Creamos el analizador de sentimientos
  vader = SentimentIntensityAnalyzer()
  #Analizamos cada frase
  for index,row in df.iterrows():
    sentiment = vader.polarity_scores(row['EN_Opiniones'])
    print(index, '---',sentiment)
    if sentiment['compound']>0.3:
      vader_clasif[index]='POS'
    elif sentiment['compound']<0.3 and sentiment['compound']>0:
      vader_clasif[index]='NEU'
    else:
      vader_clasif[index]='NEG'

  frame=pd.DataFrame( {'Clasificacion_Vader': vader_clasif})
  #final=pd.concat([df,fr_vader], axis=1)
  frame.to_excel('score_vader.xlsx')

def AsentClasificator(data:str, ColumnName:str, C=True):
  if C:
    df=TraduccionText(data,ColumnName)
  else:
      df = pd.read_excel('score_en.xlsx', sheet_name='Sheet1')

  asentimiento_clasif={}
  # load spacy pipeline
  nlp = spacy.blank('en')
  nlp.add_pipe('sentencizer')

  # add the rule-based sentiment model
  nlp.add_pipe('asent_en_v1')


  for index, row in df.iterrows():
      sentiment=nlp(row['EN_Opiniones'])#clasifica el texto
      asentimiento_clasif[index]=sentiment._.polarity   #me quede aqui
      print(asentimiento_clasif[index])
      print(index)
      if asentimiento_clasif[index].compound >0.2:
        asentimiento_clasif[index]='POS'
      elif asentimiento_clasif[index].compound<0.2 and asentimiento_clasif[index].compound>0 :
        asentimiento_clasif[index]='NEU'
      else:
        asentimiento_clasif[index]='NEG'

  frame=pd.DataFrame( {'Clasificacion_Asentiment': asentimiento_clasif})   #crear nuevo dataframe con una columna de clasificacion
  #final=pd.concat([df,fr_asent], axis=1)
  frame.to_excel('score_asent.xlsx')
