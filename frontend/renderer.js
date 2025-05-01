const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const context = canvas.getContext("2d");

const socket = new WebSocket("ws://localhost:8765");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    console.error("Error accessing webcam:", err);
  });

socket.onmessage = event => {
  const data = JSON.parse(event.data);

  // Adjust canvas to video size
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Clear the canvas
  context.clearRect(0, 0, canvas.width, canvas.height);

  if (data.faces && data.faces.length > 0) {
    for (const face of data.faces) {
      context.strokeStyle = "#00FF00";
      context.lineWidth = 3;
      context.strokeRect(face.left, face.top, face.right - face.left, face.bottom - face.top);

      context.font = "16px Arial";
      context.fillStyle = "lime";
      context.fillText(face.name, face.left, face.top - 10);
    }
  }
};

socket.onerror = (err) => {
  console.error("WebSocket error:", err);
};

socket.onclose = () => {
  console.warn("WebSocket connection closed.");
};
