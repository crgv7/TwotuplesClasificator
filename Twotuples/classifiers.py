import pandas as pd
from pysentimiento import create_analyzer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import asent

# Detectar si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de procesamiento detectado: {device}")

def PysentimentClasificator(texts: list) -> list:
    print("[Pysentimiento] Iniciando carga de modelo...")
    analyzer = create_analyzer(task="sentiment", lang="es")
    
    # Cuantización para CPU: Transforma matemática de 32bits a 8bits (Doble de rápido)
    if device.type == 'cpu':
        print("[Pysentimiento] Aplicando Cuantización de 8-bits (Turbo para CPU)...")
        analyzer.model = torch.quantization.quantize_dynamic(
            analyzer.model, {nn.Linear}, dtype=torch.qint8
        )
        
    resultados = []
    batch_size = 64
    total = len(texts)
    
    # Procesamiento por lotes
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        try:
            # Pysentimiento en versiones recientes acepta listas
            outs = analyzer.predict(batch)
            if isinstance(outs, list):
                resultados.extend([out.output for out in outs])
            else:
                resultados.append(outs.output)
        except Exception:
            # Fallback si falla el lote
            for text in batch:
                out = analyzer.predict(text)
                resultados.append(out.output)
                
        # Log de progreso
        procesados = min(i + batch_size, total)
        if procesados % (batch_size * 5) == 0 or procesados == total:
            print(f"[Pysentimiento] Progreso: {procesados}/{total} ({(procesados/total)*100:.1f}%)")
    
    print("[Pysentimiento] ¡Finalizado!")
    return resultados

def BertClasificator(texts: list) -> list:
    print("[BERT] Iniciando carga de modelo Multilingual...")
    model_id = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    # Mover modelo a la GPU si está disponible
    model.to(device)
    model.eval()
    
    # Cuantización para CPU
    if device.type == 'cpu':
        print("[BERT] Aplicando Cuantización de 8-bits (Turbo para CPU)...")
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    
    resultados = []
    batch_size = 64
    total = len(texts)
    
    # Procesamiento por lotes
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        
        # Tokenización
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Mover datos a la GPU (si aplica)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inferencia
        with torch.no_grad():
            outputs = model(**inputs)
            
        predictions = torch.argmax(outputs.logits, dim=-1)
        estrellas = predictions + 1
        
        for est in estrellas:
            e = est.item()
            if e in [1, 2]:
                resultados.append('NEG')
            elif e == 3:
                resultados.append('NEU')
            else:
                resultados.append('POS')
                
        # Log de progreso
        procesados = min(i + batch_size, total)
        if procesados % (batch_size * 5) == 0 or procesados == total:
            print(f"[BERT] Progreso: {procesados}/{total} ({(procesados/total)*100:.1f}%)")
                
    print("[BERT] ¡Finalizado!")
    return resultados

def AsentClasificator(texts: list, C=True) -> list:
    print("[Asentimiento] Iniciando carga de SpaCy...")
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('asent_en_v1')
    
    resultados = []
    total = len(texts)
    procesados = 0
    
    # spaCy usa nlp.pipe para procesar textos rápidamente en lotes
    for doc in nlp.pipe(texts, batch_size=128):
        compound = doc._.polarity.compound
        if compound > 0.2:
            resultados.append('POS')
        elif compound < 0.2 and compound > 0:
            resultados.append('NEU')
        else:
            resultados.append('NEG')
            
        procesados += 1
        # Log de progreso
        if procesados % 500 == 0 or procesados == total:
            print(f"[Asentimiento] Progreso: {procesados}/{total} ({(procesados/total)*100:.1f}%)")
            
    print("[Asentimiento] ¡Finalizado!")
    return resultados
