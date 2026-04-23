import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

def ExcelConcat():
  # Obtener una lista de todos los archivos de Excel en el directorio actual
  archivos_excel = glob.glob('*.xlsx')

  # Leer cada archivo de Excel en un DataFrame y agregarlo a una lista
  dfs = []
  for archivo in archivos_excel:
    df = pd.read_excel(archivo)
    dfs.append(df)

  # Concatenar los DataFrames en uno solo
  df_final = pd.concat(dfs,axis=1)
  return df_final

def score(df:pd):
  score=df
  score['Clasificacion_Difusa']=score['Clasicacion_Difusa'].replace([4,3,2,1,0],['POS','POS','NEU','NEG','NEG'])
  score['clasificacion_pysentimiento']=score['clasificacion_pysentimiento'].replace([2,1,0],['POS','NEU','NEG'])
  score['Clasificacion_Bert']=score['Clasificacion_Bert'].replace([2,1,0],['POS','NEU','NEG'])
  score['Clasificacion_Asentiment']=score['Clasificacion_Asentiment'].replace([2,1,0],['POS','NEU','NEG'])
  score.to_excel('score.xlsx')

def Metric(etiqueta:str, metric:str='ClassificationReport', sorter:str='difuse', ClassNumber:int=3 ):
  df=pd.read_excel('score_diffuse.xlsx', sheet_name='Sheet1')
  df[etiqueta]=df[etiqueta].replace([5,4,3,2,1],[2,2,1,0,0])
  df['Clasicacion_Difusa']=df['Clasicacion_Difusa'].replace([4,3,2,1,0],[2,2,1,0,0])


  def ConfusionMatrix(df:pd, etiqueta:str, sorter:str='difuse', ClassNumber:int=3):
    # Generar datos de ejemplo de etiquetas reales y predichas
    print('hola', ClassNumber)
    actual = df[etiqueta]
    if sorter=='pysentiment' :
        predicted = df['clasificacion_pysentimiento']
    elif sorter=='bert':
      predicted = df['Clasificacion_Bert']
    elif  sorter=='asent':
      predicted = df['Clasificacion_Asentiment']
    else:
        predicted = df['Clasicacion_Difusa']
    # Crear la matriz de confusión utilizando sklearn
    confusion_matrix_obj = metrics.confusion_matrix(actual, predicted)
    # Crear la visualización de la matriz de confusión
    if ClassNumber==2:
      cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_obj, display_labels = ['negativo','positivo'])
    else:
      cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_obj, display_labels = ['negativo','neutral','positivo'])
    # Graficar la matriz de confusión
    cm_display.plot()
    plt.show()

  def ClassificationReport(df:pd, etiqueta:str, sorter='difuse', ClassNumber=3):
    actual = df[etiqueta]
    if sorter=='pysentiment' :
        predicted = df['clasificacion_pysentimiento']
    elif sorter=='bert':
      predicted = df['Clasificacion_Bert']
    elif  sorter=='asent':
      predicted = df['Clasificacion_Asentiment']
    else:
        predicted = df['Clasicacion_Difusa']

    if ClassNumber==2:
      target_names = ['class 0', 'class 2']
    else:
      target_names = ['class 0','class 1', 'class 2'] # Fixed typo: added missing comma in target_names
    print(classification_report(actual,predicted, target_names=target_names, digits=4))


  if metric=='ConfusionMatrix':
      ConfusionMatrix(df,etiqueta, sorter, ClassNumber)
  else:
    ClassificationReport(df, etiqueta,sorter, ClassNumber)
