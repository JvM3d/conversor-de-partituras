#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import subprocess
from pdf2image import convert_from_path
import pyttsx3
from pydub import AudioSegment
from music21 import converter, midi, meter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid

# Configurações via variável de ambiente
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # ex.: "http://example.com,http://localhost:3000"
SOUNDFONT_PATH = os.getenv("SOUNDFONT_PATH", "soundfont.sf2")

app = FastAPI(
    title="Audiobook de Partituras",
    description="API para converter partituras em audiolivros narrados, acessível para deficientes visuais."
)

# CORS: em produção, recomenda-se restringir os origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "output_audio"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# Disponibiliza os arquivos gerados via URL
app.mount("/audiobooks", StaticFiles(directory=OUTPUT_DIR), name="audiobooks")

def is_sheet_music(image):
    """
    Verifica se a imagem possui características de partitura,
    detectando pautas através da análise de linhas horizontais.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    line_pixels = cv2.countNonZero(detected_lines)
    total_pixels = image.shape[0] * image.shape[1]
    return line_pixels / total_pixels > 0.01

def generate_narration_text(score, title):
    """
    Extrai informações da partitura usando music21 e gera um texto
    de narração com detalhes sobre a peça.
    """
    narration = f"A seguir, você ouvirá a peça {title}. "
    # Extrair tonalidade
    try:
        analyzed_key = score.analyze('key')
        narration += f"Esta peça está na tonalidade de {analyzed_key.tonic.name} {analyzed_key.mode}. "
    except Exception:
        narration += "A tonalidade da peça não pôde ser determinada. "
    
    # Extrair compasso
    try:
        ts = score.recurse().getElementsByClass(meter.TimeSignature)
        if ts:
            narration += f"O compasso é {ts[0].ratioString}. "
        else:
            narration += "O compasso não foi identificado. "
    except Exception:
        narration += "O compasso não foi identificado. "
    
    # Extrair andamento (bpm) via metronome marks
    try:
        tempo = None
        for mm in score.recurse().getElementsByClass('MetronomeMark'):
            tempo = mm.number
            break
        if tempo:
            narration += f"O andamento é de {tempo} batidas por minuto. "
        else:
            narration += "O andamento não foi identificado. "
    except Exception:
        narration += "O andamento não foi identificado. "
    
    narration += "Ouça atentamente e observe os detalhes para aprender a tocar esta música."
    return narration

def generate_narration_audio(narration_text, narration_file):
    """
    Converte o texto de narração em áudio utilizando pyttsx3.
    Nota: Em ambiente de produção, considere rodar esse processo de forma assíncrona.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.save_to_file(narration_text, narration_file)
    engine.runAndWait()

def find_musicxml_file(base_path):
    """
    Após chamar o Audiveris, tenta localizar o arquivo MusicXML gerado.
    Verifica tanto a extensão .xml quanto .mxl, baseando-se no nome base da imagem.
    """
    possible_xml = base_path + ".xml"
    possible_mxl = base_path + ".mxl"
    if os.path.exists(possible_xml):
        return possible_xml
    elif os.path.exists(possible_mxl):
        return possible_mxl
    return None

def process_pdf(pdf_path, output_dir=OUTPUT_DIR):
    """
    Processa o PDF de partituras:
      - Converte cada página em imagem.
      - Identifica páginas com partitura.
      - Invoca o Audiveris para gerar o MusicXML.
      - Converte o MusicXML em MIDI e, em seguida, em áudio usando FluidSynth.
      - Gera uma narração com informações musicais e combina com o áudio da peça.
      - Salva o arquivo final na pasta output_audio.
    Retorna um dicionário mapeando o título à filename do áudio final.
    """
    # Verifica se o arquivo soundfont existe
    if not os.path.exists(SOUNDFONT_PATH):
        raise Exception(f"SoundFont não encontrado em {SOUNDFONT_PATH}.")
        
    pages = convert_from_path(pdf_path, dpi=200)
    narrated_scores = {}
    
    for i, page in enumerate(pages):
        temp_id = uuid.uuid4().hex
        image_filename = f"temp_page_{temp_id}_{i}.png"
        page.save(image_filename, "PNG")
        img = cv2.imread(image_filename)
        if is_sheet_music(img):
            print(f"Página {i} identificada como partitura.")
            # Chama o Audiveris para gerar o MusicXML (certifique-se de que o Audiveris esteja instalado e no PATH)
            # Usando o nome base da imagem para que o arquivo de saída possua o mesmo nome base.
            base_name, _ = os.path.splitext(image_filename)
            cmd = f"audiveris -batch -export -output . {image_filename}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as err:
                print(f"Erro na execução do Audiveris na página {i}: {err}")
                os.remove(image_filename)
                continue

            musicxml_file = find_musicxml_file(base_name)
            if musicxml_file is None:
                print(f"Arquivo MusicXML não gerado para a página {i}.")
                os.remove(image_filename)
                continue

            try:
                score = converter.parse(musicxml_file)
                # Define o título usando metadados, ou gera um nome padrão se não houver
                if score.metadata is not None and score.metadata.title:
                    title = score.metadata.title.strip().replace(" ", "_")
                else:
                    title = f"Partitura_{temp_id}_{i}"
                
                midi_file = f"temp_page_{temp_id}_{i}.mid"
                mf = midi.translate.music21ObjectToMidiFile(score)
                mf.open(midi_file, 'wb')
                mf.write()
                mf.close()
                
                # Converte MIDI para áudio com FluidSynth
                piece_audio_file = f"temp_piece_{temp_id}_{i}.wav"
                cmd_audio = f"fluidsynth -ni {SOUNDFONT_PATH} {midi_file} -F {piece_audio_file} -r 44100"
                try:
                    subprocess.run(cmd_audio, shell=True, check=True)
                except subprocess.CalledProcessError as err:
                    print(f"Erro na conversão MIDI para áudio na página {i}: {err}")
                    os.remove(image_filename)
                    continue
                
                # Gera a narração baseada nos metadados da partitura
                narration_text = generate_narration_text(score, title)
                narration_temp_file = f"temp_narration_{temp_id}_{i}.wav"
                generate_narration_audio(narration_text, narration_temp_file)
                
                narration_audio = AudioSegment.from_wav(narration_temp_file)
                piece_audio = AudioSegment.from_wav(piece_audio_file)
                pause = AudioSegment.silent(duration=1000)  # 1 segundo de pausa
                final_audio = narration_audio + pause + piece_audio
                
                final_audio_filename = f"{title}_narrado.wav"
                final_audio_file = os.path.join(output_dir, final_audio_filename)
                final_audio.export(final_audio_file, format="wav")
                narrated_scores[title] = final_audio_filename
                print(f"Audiobook narrado salvo: {final_audio_file}")
            except Exception as e:
                print(f"Erro ao processar a página {i}: {e}")
        else:
            print(f"Página {i} não contém partitura.")
        
        # Remove arquivos temporários (se existirem)
        temp_files = [
            image_filename,
            f"temp_page_{temp_id}_{i}.xml",
            f"temp_page_{temp_id}_{i}.mxl",
            f"temp_page_{temp_id}_{i}.mid",
            f"temp_piece_{temp_id}_{i}.wav",
            f"temp_narration_{temp_id}_{i}.wav"
        ]
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
    
    return narrated_scores

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Audiobooks de Partituras. Utilize o endpoint /process para enviar um PDF."}

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """
    Endpoint para upload de PDF.
    Recebe um PDF, processa-o para gerar os audiobooks narrados e retorna os títulos e URLs de download.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="O arquivo enviado deve ser um PDF.")
    
    temp_pdf_path = f"temp_{uuid.uuid4().hex}.pdf"
    try:
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        narrated_scores = process_pdf(temp_pdf_path)
        response_data = []
        for title, filename in narrated_scores.items():
            response_data.append({
                "title": title,
                "download_url": f"{BASE_URL}/audiobooks/{filename}"
            })
        return JSONResponse(content={"audiobooks": response_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo: {e}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@app.get("/audiobooks")
def list_audiobooks():
    """
    Lista todos os audiobooks presentes na pasta de saída.
    """
    files = os.listdir(OUTPUT_DIR)
    audiobooks = [{"title": os.path.splitext(f)[0], "download_url": f"/audiobooks/{f}"} for f in files if f.endswith(".wav")]
    return JSONResponse(content={"audiobooks": audiobooks})

@app.get("/audiobooks/{filename}")
def get_audiobook(filename: str):
    """
    Endpoint para download de um audiobook específico.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, media_type="audio/wav", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Audiobook não encontrado.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
