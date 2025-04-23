# 🧠 Reconhecimento Facial com Dashboard Interativo

Sistema de reconhecimento facial em tempo real com visual corporativo, ideal para controle de acesso, demonstrações de IA ou estudos em visão computacional.

## 🎯 Funcionalidades

- 📸 Detecção facial em tempo real pela webcam
- 🔍 Reconhecimento de usuários cadastrados
- 🧱 Interface com estilo dashboard:
  - Borda animada no rosto (verde, vermelho ou azul)
  - Balão com informações do usuário (nome, idade, profissão)
  - Status visual no topo (✔ Acesso Liberado | ✖ Acesso Bloqueado | 🔎 Analisando)
- 🌐 Comunicação com WebSocket
- 🛠 Painel administrativo via navegador para editar/remover usuários

---

## 🧰 Tecnologias Usadas

- Python 3.8+
- OpenCV
- dlib
- face_recognition
- asyncio + websockets
- Flask (admin dashboard)

---

## 📂 Estrutura do Projeto

```
camera-ia-app/
├── backend/
│   ├── server.py                # Reconhecimento facial em tempo real
│   ├── salvar_rosto.py         # Cadastro de novos rostos
│   ├── face_data.json          # Dados dos usuários com encoding facial
│   ├── shape_predictor_68_face_landmarks.dat
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   └── admin/
│       ├── admin_server.py     # API Flask para gerenciar os usuários
│       ├── index.html          # Interface de gerenciamento
│       └── styles.css
└── README.md
```

---

## 🚀 Como Executar

### 1. Clone o repositório

git clone https://github.com/seu-usuario/seu-projeto.git

### 2. Instale as dependências

pip install -r requirements.txt

> ⚠️ Se houver erro com `dlib`, siga o [guia oficial de instalação](https://github.com/ageitgey/face_recognition).

### 3. Baixe os modelos obrigatórios

- [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [dlib_face_recognition_resnet_model_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

Descompacte e mova para `backend/`.

### 4. Cadastre seu rosto

```
python salvar_rosto.py
```

Digite suas informações, posicione o rosto na câmera e pressione `c` para capturar.

### 5. Inicie o reconhecimento facial

```
python server.py
```

### 6. (Opcional) Use o painel de gerenciamento

```
cd admin
python admin_server.py
```

Abra [http://localhost:5000](http://localhost:5000) no navegador.

---

## 🧪 Exemplo de Interface

📷 O rosto é detectado com um quadrado animado, e as informações aparecem em um balão com destaque. A barra superior exibe o status do acesso.

---

## 📌 Sobre os Dados

Os usuários são armazenados no arquivo `face_data.json` com:
- Nome
- Idade
- Profissão
- Encoding facial

---

## 💡 Possíveis Melhorias

- Armazenamento com banco de dados
- Login e permissões por usuário
- Deploy com Docker ou serverless
- Integração com APIs de segurança/controle de entrada

---

## 👨‍💻 Autor

Feito com 💻 e dedicação por **Allison Joanine de Araujo Ribeiro**  
📧 allisonjoanine@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/allisonjoanine) • [GitHub](https://github.com/AllisonJoanine)

---

## 📄 Licença

Este projeto é de uso livre para fins educacionais e experimentais.
```

---

# by Allison Joanine de Araujo Ribeiro
