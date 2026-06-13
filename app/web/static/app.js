const mediaInput = document.getElementById("mediaInput");
const chooseMediaBtn = document.getElementById("chooseMediaBtn");
const selectedFileName = document.getElementById("selectedFileName");
const sampleFpsInput = document.getElementById("sampleFps");
const processBtn = document.getElementById("processBtn");
const statusBadge = document.getElementById("statusBadge");
const frameCounter = document.getElementById("frameCounter");
const responseBox = document.getElementById("responseBox");
const previewCanvas = document.getElementById("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");

let isProcessing = false;

processBtn.disabled = true;

chooseMediaBtn.addEventListener("click", () => {
    mediaInput.click();
});

mediaInput.addEventListener("change", () => {
    const mediaFile = mediaInput.files?.[0];
    if (!mediaFile) {
        selectedFileName.textContent = "Файл не выбран";
        processBtn.disabled = true;
        setStatus("Ожидание файла", "idle");
        return;
    }

    const mediaKind = detectMediaKind(mediaFile);
    const sizeMb = (mediaFile.size / 1024 / 1024).toFixed(1);
    selectedFileName.textContent = `Выбран файл: ${mediaFile.name} · ${sizeMb} МБ`;
    processBtn.disabled = mediaKind === "unknown";

    if (mediaKind === "video") {
        setStatus("Видео будет обработано на backend через OpenCV", "idle");
    } else if (mediaKind === "image") {
        setStatus("Файл выбран, нажмите «Запустить обработку»", "idle");
    } else {
        setStatus("Поддерживаются изображения и видеофайлы", "error");
    }
});

processBtn.addEventListener("click", () => {
    void processMedia();
});

async function processMedia() {
    const mediaFile = mediaInput.files?.[0];
    if (!mediaFile) {
        setStatus("Сначала выберите файл", "error");
        return;
    }

    if (isProcessing) {
        return;
    }

    isProcessing = true;
    processBtn.disabled = true;
    responseBox.textContent = "Обработка...";

    try {
        const mediaKind = detectMediaKind(mediaFile);
        if (mediaKind === "image") {
            await processImage(mediaFile);
        } else if (mediaKind === "video") {
            await processVideo(mediaFile);
        } else {
            throw new Error("Поддерживаются только image/* и video/* файлы");
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : "Неизвестная ошибка";
        setStatus(`Ошибка: ${message}`, "error");
        responseBox.textContent = String(message);
    } finally {
        isProcessing = false;
        processBtn.disabled = !mediaInput.files?.[0];
    }
}

async function processImage(imageFile) {
    setStatus("Отправка изображения на backend", "busy");
    const result = await sendFrame(imageFile, 0);
    await drawFrameAndMasks(result);
    frameCounter.textContent = "Кадр: 0";
    responseBox.textContent = formatResponse(result);
    const doneStatus = result.violation_detected
        ? "Изображение обработано, нарушение найдено"
        : "Изображение обработано";
    setStatus(doneStatus, "done");
}

async function processVideo(videoFile) {
    const sampleFps = normalizeFps(Number(sampleFpsInput.value));

    setStatus("Загрузка видео на backend", "busy");
    frameCounter.textContent = "Кадр: -";

    const result = await sendVideo(videoFile, sampleFps);
    const frames = result.results || [];
    if (!frames.length) {
        throw new Error("Backend не вернул обработанные кадры");
    }

    for (let index = 0; index < frames.length; index += 1) {
        const frameResult = frames[index];
        await drawFrameAndMasks(frameResult);
        frameCounter.textContent = `Кадр: ${frameResult.frame_index}`;
        responseBox.textContent = formatVideoResponse(result, frameResult);
        await nextAnimationFrame();
    }

    const violationCount = frames.filter((frame) => frame.violation_detected).length;
    const truncatedText = result.truncated ? " Лимит кадров достигнут." : "";
    setStatus(
        `Видео обработано: ${frames.length} кадров, нарушений: ${violationCount}.${truncatedText}`,
        violationCount > 0 ? "done-warning" : "done",
    );
}

async function sendFrame(frameBlob, frameIndex) {
    const formData = new FormData();
    formData.append("frame", frameBlob, `frame_${frameIndex}.jpg`);
    formData.append("frame_index", String(frameIndex));

    const response = await fetch("/api/infer/violation", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const message = await response.text();
        throw new Error(`backend ${response.status}: ${message}`);
    }

    return response.json();
}

async function sendVideo(videoFile, sampleFps) {
    const formData = new FormData();
    formData.append("video", videoFile, videoFile.name || "video.mp4");
    formData.append("sample_fps", String(sampleFps));
    formData.append("max_frames", "120");

    const response = await fetch("/api/infer/video", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const message = await response.text();
        throw new Error(`backend ${response.status}: ${message}`);
    }

    return response.json();
}

async function drawFrameAndMasks(result) {
    const image = await loadImage(result.frame_data_url);

    configurePreviewCanvas(result.frame_width, result.frame_height);

    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.drawImage(image, 0, 0, previewCanvas.width, previewCanvas.height);

    for (const mask of result.masks || []) {
        drawMaskPolygon(mask, previewCanvas.width, previewCanvas.height);
    }

    if (result.violation_detected) {
        for (const region of result.violation_regions || []) {
            drawViolationRegion(region, previewCanvas.width, previewCanvas.height);
        }
    }
}

function configurePreviewCanvas(width, height) {
    previewCanvas.width = width;
    previewCanvas.height = height;
    previewCanvas.style.aspectRatio = `${width} / ${height}`;
    previewCanvas.setAttribute("aria-label", `Кадр ${width} на ${height}`);
}

function drawMaskPolygon(mask, width, height) {
    const points = mask.points || [];
    if (points.length < 3) {
        return;
    }

    const color = getMaskColor(mask);
    previewCtx.save();
    previewCtx.lineWidth = mask.class_id === 3 ? 4 : 3;
    previewCtx.strokeStyle = color;
    previewCtx.fillStyle = `${color}2b`;

    previewCtx.beginPath();
    previewCtx.moveTo(points[0][0] * width, points[0][1] * height);
    for (let i = 1; i < points.length; i += 1) {
        previewCtx.lineTo(points[i][0] * width, points[i][1] * height);
    }
    previewCtx.closePath();
    previewCtx.fill();
    previewCtx.stroke();

    const labelX = clamp(points[0][0] * width + 8, 8, width - 160);
    const labelY = clamp(points[0][1] * height - 10, 18, height - 8);
    const confidence = Number.isFinite(mask.confidence)
        ? ` ${(mask.confidence * 100).toFixed(0)}%`
        : "";
    previewCtx.font = "700 15px 'IBM Plex Sans', sans-serif";
    previewCtx.fillStyle = "#101820";
    previewCtx.strokeStyle = "rgba(255,255,255,0.9)";
    previewCtx.lineWidth = 4;
    previewCtx.strokeText(`${mask.class_name}${confidence}`, labelX, labelY);
    previewCtx.fillStyle = color;
    previewCtx.fillText(`${mask.class_name}${confidence}`, labelX, labelY);
    previewCtx.restore();
}

function drawViolationRegion(region, width, height) {
    const points = region.points || [];
    if (points.length < 3) {
        return;
    }

    previewCtx.save();
    previewCtx.lineWidth = Math.max(3, Math.round(Math.min(width, height) * 0.004));
    previewCtx.strokeStyle = "#e31937";
    previewCtx.fillStyle = "rgba(227, 25, 55, 0.35)";

    previewCtx.beginPath();
    previewCtx.moveTo(points[0][0] * width, points[0][1] * height);
    for (let i = 1; i < points.length; i += 1) {
        previewCtx.lineTo(points[i][0] * width, points[i][1] * height);
    }
    previewCtx.closePath();
    previewCtx.fill();
    previewCtx.stroke();

    const labelX = points[0][0] * width + previewCtx.lineWidth * 2;
    const labelY = Math.max(
        points[0][1] * height + previewCtx.lineWidth * 8,
        previewCtx.lineWidth * 8,
    );
    const confidence = Number.isFinite(region.confidence)
        ? ` ${(region.confidence * 100).toFixed(0)}%`
        : "";
    previewCtx.font = "800 16px 'IBM Plex Sans', sans-serif";
    previewCtx.fillStyle = "#ffffff";
    previewCtx.fillText(`Нарушение${confidence}`, labelX, labelY);
    previewCtx.restore();
}

function formatVideoResponse(videoResult, frameResult) {
    const lines = [
        `video: ${videoResult.filename}`,
        `source_fps: ${videoResult.source_fps ?? "-"}`,
        `sample_fps: ${videoResult.sample_fps}`,
        `frames_processed: ${videoResult.frames_processed}`,
        `truncated: ${videoResult.truncated}`,
        "",
        "Текущий кадр:",
        formatResponse(frameResult),
    ];
    return lines.join("\n");
}

function formatResponse(result) {
    const lines = [
        `frame_index: ${result.frame_index}`,
        `resolution: ${result.frame_width}x${result.frame_height}`,
        "",
    ];

    if (typeof result.violation_detected === "boolean") {
        lines.push(`violation_detected: ${result.violation_detected}`);
        lines.push(`violation_score: ${Number(result.violation_score || 0).toFixed(4)}`);
        lines.push(`reason: ${result.reason || "-"}`);
        lines.push("");
    }

    for (const region of result.violation_regions || []) {
        lines.push(`violation_model: ${region.model_name}`);
        lines.push(`class: ${region.class_id} (${region.class_name})`);
        lines.push(`confidence: ${Number(region.confidence || 0).toFixed(4)}`);
        lines.push(`bbox_xyxy: ${region.bbox_xyxy}`);
        lines.push("");
    }

    for (const mask of result.masks || []) {
        lines.push(`model: ${mask.model_name}`);
        lines.push(`class: ${mask.class_id} (${mask.class_name})`);
        if (Number.isFinite(mask.confidence)) {
            lines.push(`confidence: ${Number(mask.confidence || 0).toFixed(4)}`);
        }
        if (mask.bbox_xyxy) {
            lines.push(`bbox_xyxy: ${mask.bbox_xyxy}`);
        }
        lines.push("");
    }

    return lines.join("\n");
}

function normalizeFps(value) {
    if (!Number.isFinite(value)) {
        return 2;
    }
    return Math.min(15, Math.max(0.5, value));
}

function detectMediaKind(file) {
    const mimeType = (file.type || "").toLowerCase();
    if (mimeType.startsWith("image/")) {
        return "image";
    }

    if (mimeType.startsWith("video/")) {
        return "video";
    }

    const fileName = (file.name || "").toLowerCase();
    if (/\.(jpg|jpeg|png|bmp|webp|gif)$/.test(fileName)) {
        return "image";
    }

    if (/\.(mp4|mov|avi|mkv|webm|m4v)$/.test(fileName)) {
        return "video";
    }

    return "unknown";
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("Не удалось декодировать кадр backend"));
        image.src = src;
    });
}

function nextAnimationFrame() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
}

function getMaskColor(mask) {
    const className = String(mask.class_name || "").toLowerCase();
    if (mask.class_id === 0 || className.includes("track")) {
        return "#0aa58f";
    }
    if (mask.class_id === 1 || className.includes("wheel")) {
        return "#f1613f";
    }
    if (mask.class_id === 3 || className.includes("car")) {
        return "#2f80ed";
    }
    return "#9b51e0";
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function setStatus(text, state) {
    statusBadge.textContent = text;
    statusBadge.className = `status status-${state}`;
}
