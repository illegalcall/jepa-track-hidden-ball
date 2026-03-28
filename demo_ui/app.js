const SAMPLE_SHEET_PATH = "/demo_cases/case_1/sheet.png";

const state = {
  source: null,
  status: null,
};

function setStatus(message, tone = "quiet") {
  const pill = document.getElementById("status-pill");
  pill.textContent = message;
  pill.className = `status-pill status-pill-${tone}`;
}

function setResultMode(message, tone = "quiet") {
  const pill = document.getElementById("result-mode");
  pill.textContent = message;
  pill.className = `status-pill status-pill-${tone}`;
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { error: text || `HTTP ${response.status}` };
  }
  if (!response.ok) {
    throw new Error(payload.error || payload.details || `Request failed: ${response.status}`);
  }
  return payload;
}

function renderProbBars(probs) {
  const container = document.getElementById("prob-bars");
  container.innerHTML = "";
  probs.forEach((prob, index) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-label">Cup ${index + 1}</span>
      <div class="prob-track"><div class="prob-fill" style="width:${Math.max(prob * 100, 2)}%"></div></div>
      <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
    `;
    container.appendChild(row);
  });
}

function showPreview(src) {
  const image = document.getElementById("preview-image");
  const empty = document.getElementById("preview-empty");
  image.src = src;
  image.hidden = false;
  empty.hidden = true;
}

function useFile(file) {
  state.source = { kind: "file", file, filename: file.name };
  document.getElementById("source-label").textContent = `Source: ${file.name}`;
  showPreview(URL.createObjectURL(file));
  setStatus("Ready to run", "ready");
}

async function useSample() {
  state.source = { kind: "sample", url: SAMPLE_SHEET_PATH, filename: "sample-sheet.png" };
  document.getElementById("source-label").textContent = "Source: built-in sample sheet";
  showPreview(SAMPLE_SHEET_PATH);
  setStatus("Sample loaded", "ready");
}

function fileToBase64(fileOrBlob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(fileOrBlob);
  });
}

async function getCurrentPayload() {
  if (!state.source) {
    throw new Error("Choose an image or load the sample sheet first.");
  }

  if (state.source.kind === "file") {
    return {
      filename: state.source.filename,
      imageBase64: await fileToBase64(state.source.file),
    };
  }

  const blob = await fetch(state.source.url).then((res) => res.blob());
  return {
    filename: state.source.filename,
    imageBase64: await fileToBase64(blob),
  };
}

function renderResult(result) {
  document.getElementById("result-empty").hidden = true;
  document.getElementById("result-card").hidden = false;
  document.getElementById("prediction-badge").textContent = result.final_predicted_cup;
  document.getElementById("prediction-text").textContent = `Cup ${result.final_predicted_cup}`;
  document.getElementById("prediction-confidence").textContent =
    `Top confidence ${(Math.max(...result.final_cup_probs) * 100).toFixed(1)}%`;
  document.getElementById("frames-value").textContent = result.sheet?.num_frames ?? result.num_frames ?? "-";
  document.getElementById("history-value").textContent = result.history_size ?? "-";
  document.getElementById("job-value").textContent = result.jobId ?? "-";
  document.getElementById("device-value").textContent = state.status?.device ?? "-";
  document.getElementById("raw-json").textContent = JSON.stringify(result, null, 2);
  renderProbBars(result.final_cup_probs || []);
  setResultMode("Prediction ready", "good");
}

async function runJepa() {
  try {
    const payload = await getCurrentPayload();
    setStatus("Uploading and running", "busy");
    setResultMode("Inference in flight", "busy");
    document.getElementById("api-label").textContent = "API: /api/run-jepa-sheet";

    const result = await fetchJSON("/api/run-jepa-sheet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    renderResult(result);
    setStatus("Run completed", "good");
  } catch (error) {
    document.getElementById("result-empty").hidden = false;
    document.getElementById("result-empty").textContent = error.message;
    document.getElementById("result-card").hidden = true;
    setStatus("Run failed", "bad");
    setResultMode("Check terminal", "bad");
  }
}

async function initStatus() {
  try {
    state.status = await fetchJSON("/api/status");
    if (state.status.mode === "local") {
      document.getElementById("api-label").textContent =
        `API: local CPU · h=${state.status.historySize}`;
    } else {
      document.getElementById("api-label").textContent =
        `API: ${state.status.remoteVmName} · ${state.status.remoteVmZone} · h=${state.status.historySize}`;
    }
  } catch (error) {
    document.getElementById("api-label").textContent = `API: ${error.message}`;
  }
}

function wireDragAndDrop() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  const openPicker = () => fileInput.click();

  dropzone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    openPicker();
  });
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dropzone-hot");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dropzone-hot"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dropzone-hot");
    const file = event.dataTransfer?.files?.[0];
    if (file) useFile(file);
  });
  fileInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) useFile(file);
  });

  document.getElementById("choose-button").addEventListener("click", openPicker);
}

async function main() {
  await initStatus();
  wireDragAndDrop();

  document.getElementById("sample-button").addEventListener("click", useSample);
  document.getElementById("run-button").addEventListener("click", runJepa);
}

main().catch((error) => {
  document.body.innerHTML = `<main class="page"><p>UI boot failed: ${error.message}</p></main>`;
});
