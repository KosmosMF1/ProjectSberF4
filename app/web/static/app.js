const mediaInput = document.getElementById("mediaInput");
const chooseMediaBtn = document.getElementById("chooseMediaBtn");
const selectedFileName = document.getElementById("selectedFileName");
const sampleFpsInput = document.getElementById("sampleFps");
const staticTrackModeInput = document.getElementById("staticTrackMode");
const trackReferenceFramesInput = document.getElementById("trackReferenceFrames");
const processBtn = document.getElementById("processBtn");
const statusBadge = document.getElementById("statusBadge");
const frameCounter = document.getElementById("frameCounter");
const responseBox = document.getElementById("responseBox");
const previewCanvas = document.getElementById("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");
const videoControls = document.getElementById("videoControls");
const prevFrameBtn = document.getElementById("prevFrameBtn");
const playPauseBtn = document.getElementById("playPauseBtn");
const nextFrameBtn = document.getElementById("nextFrameBtn");
const frameSlider = document.getElementById("frameSlider");
const playbackState = document.getElementById("playbackState");

let isProcessing = false;
let currentVideoResult = null;
let currentVideoFrames = [];
let currentVideoFrameIndex = 0;
let playbackTimer = null;

processBtn.disabled = true;
resetVideoViewer();

chooseMediaBtn.addEventListener("click", () => {
    mediaInput.click();
});

mediaInput.addEventListener("change", () => {
    const mediaFile = mediaInput.files?.[0];
    stopPlayback();
    resetVideoViewer();
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    responseBox.textContent = "Нет данных";
    frameCounter.textContent = "Кадр: -";

    if (!mediaFile) {
        selectedFileName.textContent = "Файл не выбран";
        processBtn.disabled = true;
        setStatus("Ожидание файла", "idle");
        return;
    }

    const mediaKind = detectMediaKind(mediaFile);
    const sizeMb = (mediaFile.size / 1024 / 1024).toFixed(1);
    selectedFileName.textContent = `${mediaFile.name} · ${sizeMb} МБ`;
    processBtn.disabled = mediaKind === "unknown";

    if (mediaKind === "video") {
        setStatus("Видео будет обработано на backend, затем кадры можно листать", "idle");
    } else if (mediaKind === "image") {
        setStatus("Файл выбран, нажмите «Запустить обработку»", "idle");
    } else {
        setStatus("Поддерживаются изображения и видеофайлы", "error");
    }
});

processBtn.addEventListener("click", () => {
    void processMedia();
});

prevFrameBtn.addEventListener("click", () => {
    stopPlayback();
    void renderVideoFrame(currentVideoFrameIndex - 1);
});

nextFrameBtn.addEventListener("click", () => {
    stopPlayback();
    void renderVideoFrame(currentVideoFrameIndex + 1);
});

playPauseBtn.addEventListener("click", () => {
    if (playbackTimer) {
        stopPlayback();
    } else {
        startPlayback();
    }
});

frameSlider.addEventListener("input", () => {
    stopPlayback();
    void renderVideoFrame(Number(frameSlider.value));
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
    stopPlayback();

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
    resetVideoViewer();
    setStatus("Отправка изображения на backend", "busy");
    const result = await sendFrame(imageFile, 0);
    await drawFrameAndMasks(result);
    frameCounter.textContent = "Кадр: 0";
    responseBox.textContent = formatResponse(result);
    const doneStatus = result.violation_detected
        ? "Изображение обработано, нарушение найдено"
        : "Изображение обработано";
    setStatus(doneStatus, result.violation_detected ? "done-warning" : "done");
}

async function processVideo(videoFile) {
    const sampleFps = normalizeFps(Number(sampleFpsInput.value));
    const staticTrackMode = Boolean(staticTrackModeInput?.checked);
    const referenceFrames = normalizeReferenceFrames(Number(trackReferenceFramesInput.value));

    resetVideoViewer();
    setStatus(
        staticTrackMode
            ? "Загрузка видео на backend и подбор фиксированной маски трассы"
            : "Загрузка видео на backend",
        "busy",
    );
    frameCounter.textContent = "Кадр: -";

    const result = await sendVideo(videoFile, sampleFps, staticTrackMode, referenceFrames);
    const frames = result.results || [];
    if (!frames.length) {
        throw new Error("Backend не вернул обработанные кадры");
    }

    currentVideoResult = result;
    currentVideoFrames = frames;
    currentVideoFrameIndex = 0;
    setupVideoViewer(result, frames);
    await renderVideoFrame(0);

    const violationCount = frames.filter((frame) => frame.violation_detected).length;
    const carCount = countMasks(frames, isCarMask);
    const truncatedText = result.truncated ? " Лимит кадров достигнут." : "";
    const staticTrackText = result.static_track_enabled
        ? ` Фиксированная трасса: кадр ${result.static_track_reference_frame}.`
        : "";
    setStatus(
        `Видео обработано: ${frames.length} кадров, нарушений: ${violationCount}, машин: ${carCount}.${staticTrackText}${truncatedText}`,
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

async function sendVideo(videoFile, sampleFps, staticTrackMode, referenceFrames) {
    const formData = new FormData();
    formData.append("video", videoFile, videoFile.name || "video.mp4");
    formData.append("sample_fps", String(sampleFps));
    formData.append("max_frames", "120");
    formData.append("static_track_mode", String(staticTrackMode));
    formData.append("track_reference_frames", String(referenceFrames));

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

function setupVideoViewer(videoResult, frames) {
    currentVideoResult = videoResult;
    currentVideoFrames = frames;
    currentVideoFrameIndex = 0;

    videoControls.hidden = false;
    frameSlider.min = "0";
    frameSlider.max = String(Math.max(frames.length - 1, 0));
    frameSlider.value = "0";

    updatePlaybackControls();
}

function resetVideoViewer() {
    currentVideoResult = null;
    currentVideoFrames = [];
    currentVideoFrameIndex = 0;
    stopPlayback();

    if (videoControls) {
        videoControls.hidden = true;
    }
    if (frameSlider) {
        frameSlider.min = "0";
        frameSlider.max = "0";
        frameSlider.value = "0";
    }
    if (playbackState) {
        playbackState.textContent = "0 / 0";
    }
    updatePlaybackControls();
}

async function renderVideoFrame(index) {
    if (!currentVideoFrames.length || !currentVideoResult) {
        return;
    }

    const safeIndex = clamp(Math.round(index), 0, currentVideoFrames.length - 1);
    const frameResult = currentVideoFrames[safeIndex];
    currentVideoFrameIndex = safeIndex;

    await drawFrameAndMasks(frameResult);

    const total = currentVideoFrames.length;
    frameCounter.textContent = `Кадр: ${frameResult.frame_index} · ${safeIndex + 1}/${total}`;
    responseBox.textContent = formatVideoResponse(currentVideoResult, frameResult, safeIndex);
    frameSlider.value = String(safeIndex);
    updatePlaybackControls();
}

function startPlayback() {
    if (!currentVideoFrames.length || playbackTimer) {
        return;
    }

    if (currentVideoFrameIndex >= currentVideoFrames.length - 1) {
        currentVideoFrameIndex = -1;
    }

    playPauseBtn.textContent = "Пауза";
    const playbackFps = Math.min(8, normalizeFps(Number(sampleFpsInput.value)));
    const delayMs = Math.max(120, Math.round(1000 / playbackFps));

    playbackTimer = window.setInterval(() => {
        if (currentVideoFrameIndex >= currentVideoFrames.length - 1) {
            stopPlayback();
            return;
        }
        void renderVideoFrame(currentVideoFrameIndex + 1);
    }, delayMs);
}

function stopPlayback() {
    if (playbackTimer) {
        window.clearInterval(playbackTimer);
        playbackTimer = null;
    }
    if (playPauseBtn) {
        playPauseBtn.textContent = "Воспроизвести";
    }
}

function updatePlaybackControls() {
    const hasFrames = currentVideoFrames.length > 0;
    const isFirstFrame = currentVideoFrameIndex <= 0;
    const isLastFrame = currentVideoFrameIndex >= currentVideoFrames.length - 1;

    if (prevFrameBtn) {
        prevFrameBtn.disabled = !hasFrames || isFirstFrame;
    }
    if (nextFrameBtn) {
        nextFrameBtn.disabled = !hasFrames || isLastFrame;
    }
    if (playPauseBtn) {
        playPauseBtn.disabled = !hasFrames;
        if (!playbackTimer) {
            playPauseBtn.textContent = "Воспроизвести";
        }
    }
    if (frameSlider) {
        frameSlider.disabled = !hasFrames;
    }
    if (playbackState) {
        playbackState.textContent = hasFrames
            ? `${currentVideoFrameIndex + 1} / ${currentVideoFrames.length}`
            : "0 / 0";
    }
}

async function drawFrameAndMasks(result) {
    const image = await loadImage(result.frame_data_url);

    configurePreviewCanvas(result.frame_width, result.frame_height);

    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.drawImage(image, 0, 0, previewCanvas.width, previewCanvas.height);

    const masks = result.masks || [];
    for (const mask of masks.filter((item) => !isCarMask(item))) {
        drawMaskPolygon(mask, previewCanvas.width, previewCanvas.height);
    }
    for (const mask of masks.filter(isCarMask)) {
        drawMaskPolygon(mask, previewCanvas.width, previewCanvas.height);
        drawBoundingBox(mask, previewCanvas.width, previewCanvas.height);
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
    previewCtx.lineWidth = isCarMask(mask) ? 4 : 3;
    previewCtx.strokeStyle = color;
    previewCtx.fillStyle = `${color}${isCarMask(mask) ? "1f" : "2b"}`;

    previewCtx.beginPath();
    previewCtx.moveTo(points[0][0] * width, points[0][1] * height);
    for (let i = 1; i < points.length; i += 1) {
        previewCtx.lineTo(points[i][0] * width, points[i][1] * height);
    }
    previewCtx.closePath();
    previewCtx.fill();
    previewCtx.stroke();

    const labelX = clamp(points[0][0] * width + 8, 8, width - 180);
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

function drawBoundingBox(mask, width, height) {
    if (!mask.bbox_xyxy || mask.bbox_xyxy.length < 4) {
        return;
    }
    const [x1, y1, x2, y2] = mask.bbox_xyxy;
    previewCtx.save();
    previewCtx.strokeStyle = getMaskColor(mask);
    previewCtx.lineWidth = 4;
    previewCtx.setLineDash([10, 6]);
    previewCtx.strokeRect(x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height);
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

function formatVideoResponse(videoResult, frameResult, framePosition = 0) {
    const lines = [
        `video: ${videoResult.filename}`,
        `source_fps: ${videoResult.source_fps ?? "-"}`,
        `sample_fps: ${videoResult.sample_fps}`,
        `frames_processed: ${videoResult.frames_processed}`,
        `frame_position: ${framePosition + 1}/${currentVideoFrames.length || videoResult.frames_processed}`,
        `truncated: ${videoResult.truncated}`,
        `static_track_enabled: ${videoResult.static_track_enabled}`,
        `static_track_reference_frame: ${videoResult.static_track_reference_frame ?? "-"}`,
        `static_track_reason: ${videoResult.static_track_reason || "-"}`,
        "",
        "Текущий кадр:",
        formatResponse(frameResult),
    ];
    return lines.join("\n");
}

function formatResponse(result) {
    const masks = result.masks || [];
    const trackCount = masks.filter((mask) => mask.class_id === 0).length;
    const wheelCount = masks.filter((mask) => mask.class_id === 1 || String(mask.class_name || "").toLowerCase().includes("wheel")).length;
    const carCount = masks.filter(isCarMask).length;

    const lines = [
        `frame_index: ${result.frame_index}`,
        `resolution: ${result.frame_width}x${result.frame_height}`,
        `masks_total: ${masks.length}`,
        `track_masks: ${trackCount}`,
        `wheel_masks: ${wheelCount}`,
        `car_masks: ${carCount}`,
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

    for (const mask of masks) {
        lines.push(`model: ${mask.model_name}`);
        lines.push(`class: ${mask.class_id} (${mask.class_name})`);
        if (Number.isFinite(mask.confidence)) {
            lines.push(`confidence: ${Number(mask.confidence || 0).toFixed(4)}`);
        }
        if (mask.source_frame_index !== undefined) {
            lines.push(`source_frame_index: ${mask.source_frame_index}`);
        }
        if (mask.bbox_xyxy) {
            lines.push(`bbox_xyxy: ${mask.bbox_xyxy}`);
        }
        lines.push("");
    }

    return lines.join("\n");
}

function countMasks(frames, predicate) {
    return frames.reduce((total, frame) => total + (frame.masks || []).filter(predicate).length, 0);
}

function normalizeFps(value) {
    if (!Number.isFinite(value)) {
        return 2;
    }
    return Math.min(15, Math.max(0.5, value));
}

function normalizeReferenceFrames(value) {
    if (!Number.isFinite(value)) {
        return 30;
    }
    return Math.min(120, Math.max(1, Math.round(value)));
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

function getMaskColor(mask) {
    const className = String(mask.class_name || "").toLowerCase();
    if (mask.class_id === 0 || className.includes("track") || className.includes("road")) {
        return "#00b894";
    }
    if (mask.class_id === 1 || className.includes("wheel")) {
        return "#ffb000";
    }
    if (isCarMask(mask)) {
        return "#00a8ff";
    }
    return "#a29bfe";
}

function isCarMask(mask) {
    const className = String(mask.class_name || "").toLowerCase();
    const modelName = String(mask.model_name || "").toLowerCase();
    return mask.class_id === 3 || className.includes("car") || modelName.includes("car");
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function setStatus(text, state) {
    statusBadge.textContent = text;
    statusBadge.className = `status status-${state}`;
}
