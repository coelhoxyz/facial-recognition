import cv2
import dlib
import face_recognition
import numpy as np
import asyncio
import websockets
import json
import os

# Carrega dados conhecidos
with open("face_data.json", "r") as f:
    face_data = json.load(f)

known_encodings = [np.array(p["encoding"]) for p in face_data]
known_info = [(p["nome"], p["idade"], p["profissao"]) for p in face_data]

# Modelos

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


def get_face_encodings(frame):
    encodings = []
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_small)

    for face in faces:
        shape = predictor(rgb_small, face)
        encoding = face_rec_model.compute_face_descriptor(rgb_small, shape)
        encodings.append((face, np.array(encoding)))
    return encodings, frame


async def handler(websocket, *args):
    cap = cv2.VideoCapture(0)

    status_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        encodings, frame = get_face_encodings(frame)
        rostos_json = []

        acesso_liberado = False
        status_cor = (255, 255, 0)  # Azul (analisando)
        status_icone = "🔎"
        status_texto = "ANALISANDO"

        for face, encoding in encodings:
            name = "Desconhecido"
            idade = ""
            profissao = ""
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)

            if True in matches:
                match_index = matches.index(True)
                name, idade, profissao = known_info[match_index]
                acesso_liberado = True
                status_cor = (0, 255, 0)
                status_icone = "✔"
                status_texto = "ACESSO LIBERADO"
            else:
                status_cor = (0, 0, 255)
                status_icone = "✖"
                status_texto = "ACESSO BLOQUEADO"

            x1 = face.left() * 4
            y1 = face.top() * 4
            x2 = face.right() * 4
            y2 = face.bottom() * 4

            # Borda animada piscando
            thickness = 2 + ((status_frame // 10) % 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_cor, thickness)

            # Balão de info com fundo escuro, afastado para a esquerda com espaçamento
            nome_texto = f"Nome: {name}"
            idade_texto = f"Idade: {idade} anos"
            prof_texto = f"Profissao: {profissao}"

            texts = [nome_texto, idade_texto, prof_texto] if name != "Desconhecido" else ["Desconhecido"]
            largura_texto = max([cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0] for t in texts]) + 20
            altura_texto = len(texts) * 30 + 20

            info_x = x1 - largura_texto - 20  # margem maior à esquerda do quadrado
            if info_x < 10:
                info_x = 10
            info_y = y1

            cv2.rectangle(frame, (info_x - 10, info_y - 20), (info_x + largura_texto, info_y + altura_texto), (0, 0, 0), cv2.FILLED)

            for i, texto in enumerate(texts):
                cv2.putText(frame, texto, (info_x, info_y + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_cor, 2)

            rostos_json.append({
                "top": y1,
                "right": x2,
                "bottom": y2,
                "left": x1,
                "nome": name,
                "idade": idade,
                "profissao": profissao
            })

        # Status geral no topo
        if encodings:
            mensagem = f"{status_icone}  {status_texto}"
            (text_width, text_height), _ = cv2.getTextSize(mensagem, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
            box_coords = ((10, 10), (30 + text_width, 65))
            cv2.rectangle(frame, box_coords[0], box_coords[1], status_cor, thickness=cv2.FILLED)
            cv2.putText(frame, mensagem, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Reconhecimento Facial", frame)
        status_frame += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await websocket.send(json.dumps({
            "verificado": len(encodings) > 0,
            "rostos": rostos_json
        }))

    cap.release()
    cv2.destroyAllWindows()


async def main():
    print("Servidor WebSocket iniciado na porta 8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
