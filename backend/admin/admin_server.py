from flask import Flask, jsonify, request
import json
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# Agora o JSON está na mesma pasta do admin_server.py
DATA_FILE = os.path.join(os.path.dirname(__file__), "face_data.json")

# Lê os dados do arquivo JSON
def ler_dados():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                dados = json.load(f)
                print("✅ face_data.json carregado:", dados)
                return dados
        except Exception as e:
            print("❌ Erro ao ler JSON:", e)
            return []
    else:
        print("⚠️ Arquivo face_data.json não encontrado!")
    return []

# Salva os dados no arquivo JSON
def salvar_dados(dados):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(dados, f, indent=2)
            print("💾 Dados salvos com sucesso.")
    except Exception as e:
        print("❌ Erro ao salvar JSON:", e)

# Retorna os usuários para o frontend
@app.route("/usuarios", methods=["GET"])
def listar_usuarios():
    dados = ler_dados()
    usuarios_simplificados = [
        {
            "nome": u.get("nome", ""),
            "idade": u.get("idade", ""),
            "profissao": u.get("profissao", "")
        } for u in dados if "nome" in u
    ]
    print("🔁 Enviando para o frontend:", usuarios_simplificados)
    return jsonify(usuarios_simplificados)

# Atualiza os dados dos usuários
@app.route("/usuarios", methods=["POST"])
def atualizar_usuarios():
    novos_dados = request.json
    antigos = ler_dados()

    atualizados = []
    for novo in novos_dados:
        existente = next((x for x in antigos if x.get("nome") == novo["nome"]), None)
        if existente:
            existente["idade"] = novo["idade"]
            existente["profissao"] = novo["profissao"]
            atualizados.append(existente)
        else:
            novo["encoding"] = []  # opcional: pode deixar vazio
            atualizados.append(novo)

    salvar_dados(atualizados)
    return jsonify({"mensagem": "Salvo com sucesso!"})


if __name__ == "__main__":
    print(f"🚀 Servidor iniciado. Lendo dados de: {DATA_FILE}")
    app.run(port=5000)
