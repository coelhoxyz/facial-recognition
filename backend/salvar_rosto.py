import cv2
import face_recognition
import json
import os

ARQUIVO_JSON = "face_data.json"

def carregar_base():
    if os.path.exists(ARQUIVO_JSON):
        with open(ARQUIVO_JSON, "r") as f:
            return json.load(f)
    return []

def salvar_base(base):
    with open(ARQUIVO_JSON, "w") as f:
        json.dump(base, f, indent=2)

def capturar_rosto():
    video = cv2.VideoCapture(0)
    print("[INFO] Pressione 'c' para capturar a imagem ou 'q' para sair.")

    while True:
        ret, frame = video.read()
        if not ret:
            continue

        cv2.imshow("Captura de Rosto", frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('c'):
            video.release()
            cv2.destroyAllWindows()
            return frame
        elif key & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            exit()

def processar_rosto(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)

    if len(faces) != 1:
        print("[ERRO] Nenhum rosto ou múltiplos rostos detectados. Tente novamente.")
        return None

    encoding = face_recognition.face_encodings(rgb, known_face_locations=faces)[0]
    return encoding.tolist()

def main():
    print("=== Cadastro Facial ===")
    nome = input("Nome completo: ")
    idade = input("Idade: ")
    profissao = input("Profissão: ")

    frame = capturar_rosto()
    encoding = processar_rosto(frame)

    if encoding is None:
        return

    nova_pessoa = {
        "nome": nome,
        "idade": idade,
        "profissao": profissao,
        "encoding": encoding
    }

    base = carregar_base()
    base.append(nova_pessoa)
    salvar_base(base)

    print(f"[SUCESSO] Rosto de {nome} salvo com sucesso!")

if __name__ == "__main__":
    main()
