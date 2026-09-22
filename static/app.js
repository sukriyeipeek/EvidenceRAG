const fileInput = document.getElementById("documentFile");
const dropzone = document.getElementById("dropzone");
const fileName = document.getElementById("fileName");
const uploadButton = document.getElementById("uploadButton");
const askButton = document.getElementById("askButton");
const questionInput = document.getElementById("question");
const uploadFeedback = document.getElementById("uploadFeedback");
const askFeedback = document.getElementById("askFeedback");
const answerCard = document.getElementById("answerCard");
const sourcesWrapper = document.getElementById("sourcesWrapper");
const sourcesList = document.getElementById("sourcesList");
const sourceCount = document.getElementById("sourceCount");
const chunkCount = document.getElementById("chunkCount");
const healthDot = document.getElementById("healthDot");
const healthText = document.getElementById("healthText");

fileInput.addEventListener("change", () => {
    updateSelectedFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragging");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragging");
    });
});

dropzone.addEventListener("drop", (event) => {
    const droppedFile = event.dataTransfer.files[0];
    if (!droppedFile) return;
    fileInput.files = event.dataTransfer.files;
    updateSelectedFile(droppedFile);
});

uploadButton.addEventListener("click", uploadDocument);
askButton.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        askQuestion();
    }
});

window.addEventListener("load", refreshHealth);

function updateSelectedFile(file) {
    if (!file) {
        fileName.textContent = "Dosyanı buraya bırak";
        return;
    }
    fileName.textContent = file.name;
}

async function uploadDocument() {
    const file = fileInput.files[0];
    if (!file) {
        showFeedback(uploadFeedback, "Önce bir doküman seçmelisin.", true);
        return;
    }

    setButtonLoading(uploadButton, true, "Indexleniyor...");
    clearFeedback(uploadFeedback);

    try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/documents", { method: "POST", body: formData });
        const data = await parseResponse(response);

        if (!response.ok) {
            throw new Error(data.detail || "Doküman indexlenemedi.");
        }

        chunkCount.textContent = data.chunks_indexed;
        showFeedback(
            uploadFeedback,
            `${data.filename} başarıyla indexlendi. ${data.chunks_indexed} chunk eklendi.`
        );
        await refreshHealth();
    } catch (error) {
        showFeedback(uploadFeedback, error.message, true);
    } finally {
        setButtonLoading(uploadButton, false, "Dokümanı indexle");
    }
}

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        showFeedback(askFeedback, "Lütfen bir soru yazmalısın.", true);
        return;
    }

    setButtonLoading(askButton, true, "Cevap hazırlanıyor...");
    clearFeedback(askFeedback);

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        const data = await parseResponse(response);

        if (!response.ok) {
            throw new Error(data.detail || "Cevap oluşturulamadı.");
        }

        renderAnswer(data.answer, data.sources || []);
    } catch (error) {
        showFeedback(askFeedback, error.message, true);
    } finally {
        setButtonLoading(askButton, false, "Cevap oluştur");
    }
}

async function refreshHealth() {
    try {
        const response = await fetch("/health");
        const data = await response.json();
        if (!response.ok) throw new Error();
        healthDot.classList.add("online");
        healthText.textContent = `Backend bağlı · ${data.indexed_chunks} chunk hazır`;
        chunkCount.textContent = data.indexed_chunks;
    } catch {
        healthDot.classList.remove("online");
        healthText.textContent = "Backend bağlantısı yok";
    }
}

async function parseResponse(response) {
    const data = await response.json().catch(() => ({}));
    return data;
}

function renderAnswer(answer, sources) {
    answerCard.className = "answer-card answer-ready";
    answerCard.replaceChildren();

    const icon = document.createElement("div");
    icon.className = "empty-icon";
    icon.textContent = "✦";

    const content = document.createElement("div");
    const label = document.createElement("p");
    label.className = "answer-label";
    label.textContent = "Evidence-based answer";
    const answerText = document.createElement("div");
    answerText.className = "answer-text";
    answerText.textContent = answer;
    content.append(label, answerText);
    answerCard.append(icon, content);

    renderSources(sources);
}

function renderSources(sources) {
    sourcesList.replaceChildren();

    if (!sources.length) {
        sourceCount.textContent = "Kaynak bulunamadı";
        sourcesWrapper.hidden = false;
        const empty = document.createElement("p");
        empty.className = "panel-description";
        empty.textContent = "Bu soru için yeterli relevance skoruna sahip kanıt bulunamadı.";
        sourcesList.appendChild(empty);
        return;
    }

    sourceCount.textContent = `${sources.length} kanıt parçası`;
    sourcesWrapper.hidden = false;

    sources.forEach((source, index) => {
        const card = document.createElement("article");
        card.className = "source-card";

        const top = document.createElement("div");
        top.className = "source-card-top";
        top.append(
            textElement("span", "source-number", `KANIT ${String(source.evidence_number || index + 1).padStart(2, "0")}`),
            textElement("span", "source-score", `Benzerlik ${source.score.toFixed(3)}`)
        );

        const fileRow = document.createElement("div");
        fileRow.className = "source-file-row";
        fileRow.append(
            textElement("span", "file-symbol", "▤"),
            textElement("strong", "source-filename", source.filename || "Bilinmeyen dosya")
        );

        const meta = document.createElement("div");
        meta.className = "source-meta";
        meta.append(
            textElement("span", "", "Sayfa "),
            textElement("strong", "", source.page ?? "bilinmiyor")
        );

        const evidenceText = textElement("blockquote", "source-evidence", source.text || "Bu kaynak için metin sağlanmadı.");

        const details = document.createElement("details");
        details.className = "source-details";
        details.appendChild(textElement("summary", "", "Teknik kaynak bilgisi"));
        const detailText = `chunk_id: ${source.chunk_id || "-"}\ndocument_id: ${source.document_id || "-"}\nsource: ${source.source || "-"}`;
        const detailParagraph = textElement("p", "", detailText);
        detailParagraph.style.whiteSpace = "pre-wrap";
        details.appendChild(detailParagraph);

        card.append(top, fileRow, meta, evidenceText, details);
        sourcesList.appendChild(card);
    });
}

function textElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text;
    return element;
}

function setButtonLoading(button, loading, label) {
    button.disabled = loading;
    button.querySelector("span").textContent = label;
}

function showFeedback(element, message, error = false) {
    element.textContent = message;
    element.classList.toggle("error", error);
}

function clearFeedback(element) {
    element.textContent = "";
    element.classList.remove("error");
}
