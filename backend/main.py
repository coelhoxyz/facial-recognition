import asyncio
import websockets
import cv2
import face_recognition
import os
import numpy as np
import json

def carregar_rostos():
    rostos = []
    nomes = []
    pasta = 'face_data'
    if not os.path.exists(pasta):
        print(f"[ERRO] Pasta '{pasta}' não encontrada.")
        return rostos, nomes

    for nome_arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, nome_arquivo)
        imagem = face_recognition.load_image_file(caminho)
        localizacoes = face_recognition.face_locations(imagem)
        codificacoes = face_recognition.face_encodings(imagem, localizacoes)
        if codificacoes:
            rostos.append(codificacoes[0])
            nomes.append(os.path.splitext(nome_arquivo)[0])
        else:
            print(f"[AVISO] Nenhuma face detectada em {nome_arquivo}")
    return rostos, nomes

rostos_conhecidos, nomes_conhecidos = carregar_rostos()
camera = cv2.VideoCapture(0)

async def reconhecer(websocket):
    print("[INFO] Cliente conectado.")
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                continue

            frame_pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = frame_pequeno[:, :, ::-1]

            resultados = []
            verificado = False

            try:
                localizacoes = face_recognition.face_locations(rgb)
                if not localizacoes:
                    await websocket.send(json.dumps({"rostos": [], "verificado": False}))
                    await asyncio.sleep(0.05)
                    continue

                codificacoes = face_recognition.face_encodings(rgb, localizacoes)
                for (top, right, bottom, left), codificacao in zip(localizacoes, codificacoes):
                    matches = face_recognition.compare_faces(rostos_conhecidos, codificacao)
                    nome = "Desconhecido"
                    if True in matches:
                        nome = nomes_conhecidos[matches.index(True)]
                        verificado = True

                    resultados.append({
                        "top": top * 4,
                        "right": right * 4,
                        "bottom": bottom * 4,
                        "left": left * 4,
                        "nome": nome
                    })

            except Exception as e:
                print(f"[ERRO] Falha ao detectar ou codificar rosto: {e}")

            await websocket.send(json.dumps({
                "rostos": resultados,
                "verificado": verificado
            }))
            await asyncio.sleep(0.05)
    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Cliente desconectado.")

async def main():
    async with websockets.serve(reconhecer, "localhost", 8765):
        print("[INFO] Servidor WebSocket iniciado em ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Encerrando servidor...")
        camera.release()
