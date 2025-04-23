const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const context = canvas.getContext("2d");

const socket = new WebSocket("ws://localhost:8765");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    console.error("Erro ao acessar webcam:", err);
  });

socket.onmessage = event => {
  const data = JSON.parse(event.data);

  // Ajusta o canvas ao tamanho do vídeo
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Limpa o canvas
  context.clearRect(0, 0, canvas.width, canvas.height);

  if (data.rostos && data.rostos.length > 0) {
    for (const rosto of data.rostos) {
      context.strokeStyle = "#00FF00";
      context.lineWidth = 3;
      context.strokeRect(rosto.left, rosto.top, rosto.right - rosto.left, rosto.bottom - rosto.top);

      context.font = "16px Arial";
      context.fillStyle = "lime";
      context.fillText(rosto.nome, rosto.left, rosto.top - 10);
    }
  }
};

socket.onerror = (err) => {
  console.error("Erro no WebSocket:", err);
};

socket.onclose = () => {
  console.warn("Conexão WebSocket encerrada.");
};
