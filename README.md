<div align="center">

  <h1>Análisis de Lenguaje Corporal con IA</h1>
  <p><strong>Sistema de visión artificial en tiempo real con YOLOv8 Pose y modelos multimodales (Ollama) que detecta señales no verbales y envía informes conductuales a Telegram.</strong></p>

  <p>
    <a href="https://skillicons.dev">
      <img src="https://skillicons.dev/icons?i=python,opencv,pytorch,vscode,git,github" />
    </a>
  </p>

  <p>
    <a href="#-arquitectura-del-flujo"><img src="https://img.shields.io/badge/01_ARQUITECTURA-FF6B00?style=for-the-badge" /></a>
    <a href="#-señales-detectadas"><img src="https://img.shields.io/badge/02_SEÑALES_NO_VERBALES-1E293B?style=for-the-badge" /></a>
    <a href="#-stack-técnico"><img src="https://img.shields.io/badge/03_TECNOLOGÍAS-1E293B?style=for-the-badge" /></a>
    <a href="#-despliegue-local"><img src="https://img.shields.io/badge/04_INSTALACIÓN-FF6B00?style=for-the-badge" /></a>
  </p>

</div>

---

<table>
  <tr>
    <td width="55%" valign="center">
      <h3>🎯 Visión del Proyecto</h3>
      <p>
        Sistema inteligente pensado para entornos de entrevistas y presentaciones. Utiliza <strong>YOLOv8-Pose</strong> para extraer 17 puntos anatómicos en vivo, clasifica micro-conductas mediante geometría corporal y delega en un <strong>hilo secundario (multithreading)</strong> el análisis multimodal remoto con <strong>Ollama</strong> y el envío de reportes por <strong>Telegram</strong>.
      </p>
      <ul>
        <li>🦴 <strong>Tracking esquelético (17 Keypoints):</strong> Ojos, orejas, hombros y muñecas con escala adaptativa.</li>
        <li>🧵 <strong>Arquitectura concurrente:</strong> Inferencia remota en segundo plano sin congelar los FPS de la cámara.</li>
        <li>📲 <strong>Alertas ejecutivas:</strong> Captura fotográfica + informe psicológico cada 60 segundos en Telegram.</li>
      </ul>
    </td>
    <td width="45%" align="center">
      <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaG5tZ3F4Y3Z4Y3Z4Y3Z4Y3Z4Y3Z4Y3Z4Y3Z4Y3Z4Y3Z4/L1R1tvI9svkIWwpVYr/giphy.gif" width="100%" alt="AI Pose Analysis" />
    </td>
  </tr>
</table>

---

### ⚙️ 01 Arquitectura del Flujo (Pipeline)

<div align="center">

```mermaid
flowchart LR
    A["01 Webcam Live<br/>(YOLOv8n-Pose · 17 pts)"] --> B["02 Árbol Geométrico<br/>(Escala Orejas / Hombros)"]
    B --> C["03 Buffer 60 seg<br/>(Validación ≥ 2.0s)"]
    C --> D["04 Hilo IA Remoto<br/>(Ollama Multimodal + Foto)"]
    D --> E["05 Bot Telegram<br/>(Captura + Informe HTML)"]
    
    style A fill:#161b22,stroke:#ff6b00,stroke-width:2px,color:#fff
    style B fill:#161b22,stroke:#38bdf8,stroke-width:2px,color:#fff
    style C fill:#161b22,stroke:#a855f7,stroke-width:2px,color:#fff
    style D fill:#161b22,stroke:#38bdf8,stroke-width:2px,color:#fff
    style E fill:#ff6b00,stroke:#fff,stroke-width:2px,color:#fff
