// Loads the model into the browser and labels the text a visitor types.
// Transformers.js runs the ONNX model with ONNX Runtime Web. The version is pinned so the
// page keeps working the same way.
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { normalizeText } from "./normalize.js";

const LABELS = ["negative", "neutral", "positive"];
const MAX_CHARS = 1000;

// The model files sit next to this page, so nothing is fetched from the Hub.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL(".", import.meta.url).href;

const $ = (id) => document.getElementById(id);
const textBox = $("text");
const button = $("classify");
const status = $("status");
const progress = $("progress");

let classifier;

function setBusy(busy, message) {
  textBox.disabled = busy;
  button.disabled = busy;
  document.querySelectorAll("#examples button").forEach((b) => (b.disabled = busy));
  status.textContent = message;
}

function showError(message) {
  $("result").replaceChildren(Object.assign(document.createElement("p"), { className: "error", textContent: message }));
}

function show(scores) {
  const byLabel = Object.fromEntries(scores.map((s) => [s.label, s.score]));
  const best = LABELS.reduce((a, b) => (byLabel[a] >= byLabel[b] ? a : b));
  const verdict = Object.assign(document.createElement("p"), { className: `verdict ${best}`, textContent: best });
  const bars = LABELS.map((label) => {
    const percent = (byLabel[label] * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `<span>${label}</span><span class="track"><span class="fill ${label}" style="display:block;width:${percent}%"></span></span><span class="pct">${percent}%</span>`;
    return row;
  });
  $("result").replaceChildren(verdict, ...bars);
}

async function classify() {
  const text = textBox.value;
  if (text.length > MAX_CHARS) return showError(`Please use at most ${MAX_CHARS} characters.`);
  const cleaned = normalizeText(text);
  if (!cleaned) return showError("Please write at least one word.");
  setBusy(true, "Thinking...");
  try {
    show(await classifier(cleaned, { top_k: null }));
  } finally {
    setBusy(false, "Ready");
  }
}

async function start() {
  const examples = await (await fetch(new URL("examples.json", import.meta.url))).json();
  $("examples").replaceChildren(
    ...examples.map((example) => {
      const item = document.createElement("li");
      const chip = Object.assign(document.createElement("button"), { type: "button", textContent: example, disabled: true });
      chip.addEventListener("click", () => {
        textBox.value = example;
        classify();
      });
      item.append(chip);
      return item;
    }),
  );
  fetch(new URL("model/model-info.json", import.meta.url))
    .then((r) => r.json())
    .then((info) => {
      $("model-info").innerHTML = `Model: <a href="https://huggingface.co/${info.adapter_repo}"><code>${info.adapter_repo}</code></a> at revision <code>${info.adapter_revision.slice(0, 8)}</code>.`;
    })
    .catch(() => {});

  progress.hidden = false;
  status.textContent = "Loading the model...";
  try {
    classifier = await pipeline("text-classification", "model", {
      dtype: "q8",
      progress_callback: (p) => {
        if (p.status === "progress" && p.file?.endsWith(".onnx")) progress.value = p.progress;
      },
    });
  } catch (error) {
    progress.hidden = true;
    status.textContent = "";
    return showError(`The model could not be loaded: ${error.message}`);
  }
  progress.hidden = true;
  setBusy(false, "Ready");
  button.addEventListener("click", classify);
  textBox.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) classify();
  });
}

start();
