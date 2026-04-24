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
    print("[BERT] Iniciando carga de modelo ONNX (carlosrgv/bert-sentimiento-hoteles-onnx)...")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline
    
    repo_id = "carlosrgv/bert-sentimiento-hoteles-onnx"
    
    # Cargar ONNX directamente desde Hugging Face
    model = ORTModelForSequenceClassification.from_pretrained(repo_id, file_name="model_quantized.onnx")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    resultados = []
    batch_size = 128 # La pipeline procesa muy rápido en CPU
    total = len(texts)
    
    # Procesamiento por lotes para mantener el progreso
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        # Truncar los textos a 512 para evitar errores de longitud en el tokenizador
        batch_trunc = [str(t)[:512] for t in batch]
        
        try:
            # Inferir el lote
            outs = classifier(batch_trunc)
            
            for out in outs:
                label = str(out['label']).upper()
                # Mapeo universal por si devuelve estrellas o POS/NEG directamente
                if "1" in label or "2" in label or "NEG" in label:
                    resultados.append('NEG')
                elif "3" in label or "NEU" in label:
                    resultados.append('NEU')
                else:
                    resultados.append('POS')
        except Exception as e:
            # Fallback a uno por uno si falla un lote
            for t in batch_trunc:
                try:
                    out = classifier(t)[0]
                    label = str(out['label']).upper()
                    if "1" in label or "2" in label or "NEG" in label:
                        resultados.append('NEG')
                    elif "3" in label or "NEU" in label:
                        resultados.append('NEU')
                    else:
                        resultados.append('POS')
                except Exception:
                    resultados.append('NEU')
                
        # Log de progreso
        procesados = min(i + batch_size, total)
        if procesados % (batch_size * 4) == 0 or procesados == total:
            print(f"[BERT-ONNX] Progreso: {procesados}/{total} ({(procesados/total)*100:.1f}%)")
                
    print("[BERT-ONNX] ¡Finalizado!")
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
        if compound > 0.1:
            resultados.append('POS')
        elif compound < 0.1 and compound > 0:
            resultados.append('NEU')
        else:
            resultados.append('NEG')
            
        procesados += 1
        # Log de progreso
        if procesados % 500 == 0 or procesados == total:
            print(f"[Asentimiento] Progreso: {procesados}/{total} ({(procesados/total)*100:.1f}%)")
            
    print("[Asentimiento] ¡Finalizado!")
    return resultados
