
import json
import xml.etree.ElementTree as ET
import re
import os
from tqdm import tqdm

NEGACIONES = {'no', 'nada', 'nunca', 'jamás', 'ni', 'tampoco', 'sin'}
PUNTUACION_CORTE = {'.', ',', ';', '!', '?'}
INTENSIFICADORES = {'muy', 'demasiado', 'super', 'súper', 'extremadamente', 'bastante', 'altamente', 'totalmente', 'realmente'}
ATENUADORES = {'poco', 'algo', 'ligeramente', 'apenas', 'medio'}
CONTRASTES = {'pero', 'aunque'}
def cargar_lexico_json(path):
    print(f"[Lexicon JSON] Cargando léxico JSON: {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lexicon = {str(k).lower(): float(v) for k, v in data.items()}
        
        # Omitimos las negaciones
        for neg in NEGACIONES:
            if neg in lexicon:
                del lexicon[neg]
                
        print(f"[Lexicon JSON] Léxico cargado: {len(lexicon)} términos listos.")
        return lexicon
    except Exception as e:
        print(f"[Lexicon JSON] Error al leer el archivo JSON: {e}")
        return None

def LexiconJSONClasificator(texts: list) -> list:
    path_json = os.path.join(os.path.dirname(__file__), 'lexicon.json')
    
    diccionario_json = cargar_lexico_json(path_json)
    if not diccionario_json:
        print("[Lexicon JSON] Usando diccionario vacío como fallback.")
        diccionario_json = {}
        
    resultados = []
    scores = []
    
    print(f"[Lexicon JSON] Analizando {len(texts)} textos con Léxico JSON...")
    
    for texto in tqdm(texts, desc="Progreso del análisis", unit="txt"):
        s = analizar_sentimiento_avanzado(str(texto), diccionario_json)
        scores.append(s)
        
        if s > 0.5:
            resultados.append('POS')
        elif s < -0.5:
            resultados.append('NEG')
        else:
            resultados.append('NEU')

    print("[Lexicon JSON] ¡Finalizado!")
    return resultados, scores

def SentimentAnalysisSpanish(texts: list) -> list:
    print("[Sentiment Analysis Spanish] Iniciando carga de modelo...")
    from sentiment_analysis_spanish import sentiment_analysis
    
    sentiment = sentiment_analysis.SentimentAnalysisSpanish()
    
    resultados = []
    scores = []
    
    print(f"[Sentiment Analysis Spanish] Analizando {len(texts)} textos...")
    
    for texto in tqdm(texts, desc="Progreso del análisis", unit="txt"):
        try:
            raw_val = sentiment.sentiment(str(texto))
            
            # Mapear de Probabilidad (0 a 1) a Escala de Sentimiento (-1 a 1)
            # Para que -1 sea Negativo, 0 sea Neutro y 1 sea Positivo
            val_mapped = (float(raw_val) * 2.0) - 1.0
            val = round(val_mapped, 3)
            
            scores.append(val)
            
            # Clasificación usando la nueva escala (-1 a 1)
            if val > 0.1:
                resultados.append('POS')
            elif val < -0.999:
                resultados.append('NEG')
            else:
                resultados.append('NEU')
        except Exception:
            scores.append(0.0)
            resultados.append('NEU')
                
    print("[Sentiment Analysis Spanish] ¡Finalizado!")
    return resultados, scores


def cargar_senticon_xml(path):
    print(f"[Senticon] Cargando léxico: {path}...")
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Senticon] Error al abrir el XML: {e}")
        return None

    lexicon = {}
    for lemma in root.iter('lemma'):
        palabra = lemma.text
        score_str = lemma.get('pol')
        
        if palabra and score_str:
            p_limpia = palabra.strip().lower()
            if p_limpia == 'no':
                continue
            lexicon[p_limpia] = float(score_str)

    print(f"[Senticon] Léxico cargado: {len(lexicon)} palabras (excluyendo 'no').")
    return lexicon

def analizar_sentimiento_avanzado(texto, diccionario):
    if not isinstance(texto, str) or texto.strip() == "":
        return 0.0
    
    tokens = re.findall(r'\w+|[.,!?;]', texto.lower())
    
    score_total = 0.0
    negacion_activa = False
    ventana_negacion = 0 
    
    multiplicador_intensidad = 1.0
    ventana_intensidad = 0
    
    multiplicador_contraste = 1.0

    for t in tokens:
        if t in CONTRASTES:
            # Lógica del "PERO": Reducir a la mitad el sentimiento anterior
            score_total *= 0.5
            # Y darle un 50% más de peso a todo lo que viene después (hasta el punto)
            multiplicador_contraste = 1.5
            continue

        if t in PUNTUACION_CORTE:
            negacion_activa = False
            ventana_negacion = 0
            multiplicador_contraste = 1.0 # El contraste se reinicia con la puntuación
            continue

        if t in NEGACIONES:
            negacion_activa = True
            ventana_negacion = 3 
            continue
            
        if t in INTENSIFICADORES:
            multiplicador_intensidad = 1.5
            ventana_intensidad = 2
            continue
            
        if t in ATENUADORES:
            multiplicador_intensidad = 0.5
            ventana_intensidad = 2
            continue

        if t in diccionario:
            valor = diccionario[t]
            
            # Aplicar negación
            if negacion_activa and ventana_negacion > 0:
                valor *= -1.0
                ventana_negacion -= 1
                
            # Aplicar intensificadores/atenuadores
            if ventana_intensidad > 0:
                valor *= multiplicador_intensidad
                ventana_intensidad -= 1
                
            # Aplicar contraste
            valor *= multiplicador_contraste
            
            score_total += valor
        
        # Limpiar ventanas si se acaban
        if ventana_negacion == 0:
            negacion_activa = False
        if ventana_intensidad == 0:
            multiplicador_intensidad = 1.0
            
    return score_total

def SenticonClasificator(texts: list, C=True) -> list:
    # Buscar el XML siempre dentro de la misma carpeta donde está instalado este script
    path_xml = os.path.join(os.path.dirname(__file__), 'senticon.es.xml')
            
    senticon_dict = cargar_senticon_xml(path_xml)
    if not senticon_dict:
        print("[Senticon] Usando diccionario vacío como fallback.")
        senticon_dict = {}
        
    resultados = []
    scores = []
    
    print(f"[Senticon] Analizando {len(texts)} textos con lógica de negación...")
    
    for texto in tqdm(texts, desc="Progreso del análisis", unit="txt"):
        s = analizar_sentimiento_avanzado(str(texto), senticon_dict)
        scores.append(s)
        
        if s > 0.1:
            resultados.append('POS')
        elif s < -0.1:
            resultados.append('NEG')
        else:
            resultados.append('NEU')

    print("[Senticon] ¡Finalizado!")
    return resultados, scores
