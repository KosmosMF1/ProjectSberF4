(() => {
    const dialog = document.getElementById("calibrationDialog");
    const title = document.getElementById("calibrationTitle");
    const canvas = document.getElementById("calibrationCanvas");
    const ctx = canvas.getContext("2d");
    const status = document.getElementById("calibrationStatus");
    const time = document.getElementById("calibrationTime");
    const frameTime = document.getElementById("calibrationFrameTime");
    const previewForm = document.getElementById("calibrationPreviewForm");
    const previewBtn = document.getElementById("calibrationPreviewBtn");
    const closeBtn = document.getElementById("closeCalibrationBtn");
    const saveBtn = document.getElementById("saveCalibrationBtn");
    const undoBtn = document.getElementById("undoCalibrationPoint");
    const clearBtn = document.getElementById("clearCalibrationPoints");

    let draft = null;
    let busy = false;

    function refreshControls() {
        previewBtn.disabled = time.disabled = busy;
        undoBtn.disabled = clearBtn.disabled = busy || !draft?.points.length;
        saveBtn.disabled = busy || !draft?.image || draft.points.length < 3;
    }

    function refreshVideoOptions() {
        const videoSelected = detectMediaKind(mediaInput.files?.[0]) === "video";
        openCalibrationBtn.disabled = !videoSelected || isProcessing;
        useCameraCalibrationInput.disabled = !videoSelected;
    }

    mediaInput.addEventListener("change", () => {
        useCameraCalibrationInput.checked = false;
        refreshVideoOptions();
    });
    cameraIdInput.addEventListener("input", () => {
        useCameraCalibrationInput.checked = false;
    });
    closeBtn.addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
        draft?.controller.abort();
        draft = null;
        busy = false;
    });

    function draw() {
        if (!draft?.image) return;
        const { points, image } = draft;
        const { width, height } = canvas;
        ctx.drawImage(image, 0, 0, width, height);

        if (points.length) {
            ctx.beginPath();
            points.forEach(([x, y], i) => {
                if (i === 0) ctx.moveTo(x * width, y * height);
                else ctx.lineTo(x * width, y * height);
            });
            if (points.length >= 3) {
                ctx.closePath();
                ctx.fillStyle = "rgba(0, 184, 148, 0.20)";
                ctx.fill();
            }
            ctx.strokeStyle = "#60ffd5";
            ctx.lineWidth = 3;
            ctx.stroke();

            for (const [x, y] of points) {
                ctx.beginPath();
                ctx.arc(x * width, y * height, 4, 0, 2 * Math.PI);
                ctx.fillStyle = "#ffffff";
                ctx.fill();
            }
        }
    }

    async function loadPreview() {
        const active = draft;
        const request = new FormData();
        request.append("video", active.file, active.file.name);
        request.append("time_seconds", time.value);
        const response = await fetch("/api/calibration/preview", {
            method: "POST",
            body: request,
            signal: active.controller.signal,
        });
        if (!response.ok) throw new Error(await readBackendError(response));
        const frame = await response.json();
        const image = await loadImage(frame.frame_data_url);
        if (draft !== active || active.controller.signal.aborted) return;

        const firstPreview = !active.frame;
        const previous = active.frame || active.profile;
        const incompatible = previous
            && Math.abs(frame.frame_width * previous.frame_height / (frame.frame_height * previous.frame_width) - 1) > 0.005;
        if (incompatible) draft.points = [];
        draft.frame = frame;
        draft.image = image;
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        canvas.hidden = false;
        frameTime.textContent = `Кадр на ${frame.timestamp.toFixed(2)} с`;
        if (frame.last_timestamp !== null) {
            time.max = String(frame.last_timestamp);
        }
        status.textContent = incompatible
            ? "Пропорции кадра отличаются. Разметь границу для этого кадра заново."
            : firstPreview && active.profile ? "Показана сохранённая граница." : "";
        draw();
    }

    async function run(action, message) {
        const active = draft;
        busy = true;
        status.textContent = message;
        refreshControls();
        try {
            await action();
        } catch (error) {
            if (draft === active && error.name !== "AbortError") {
                status.textContent = error.message;
            }
        } finally {
            if (draft === active) {
                busy = false;
                refreshControls();
            }
        }
    }

    openCalibrationBtn.addEventListener("click", () => {
        const file = mediaInput.files?.[0];
        if (detectMediaKind(file) !== "video" || !cameraIdInput.reportValidity()) return;
        const cameraId = cameraIdInput.value.trim();
        if (!cameraId) {
            setStatus("Укажите ID камеры для сохранения границы", "error");
            return;
        }
        draft = {
            file, cameraId, path: `/api/cameras/${encodeURIComponent(cameraId)}/calibration`,
            controller: new AbortController(), points: [], frame: null, image: null,
        };
        title.textContent = `Граница трассы · ${cameraId}`;
        canvas.hidden = true;
        time.value = "0";
        time.removeAttribute("max");
        frameTime.textContent = "";
        dialog.showModal();
        void run(async () => {
            const active = draft;
            const response = await fetch(active.path, { signal: active.controller.signal });
            let profile = null;
            if (response.ok) profile = await response.json();
            else if (response.status !== 404) throw new Error(await readBackendError(response));
            if (draft !== active) return;
            active.profile = profile;
            if (profile) active.points = profile.points;
            await loadPreview();
        }, "Загружается кадр для разметки…");
    });

    previewForm.addEventListener("submit", (event) => {
        event.preventDefault();
        if (!busy) void run(loadPreview, "Загружается кадр…");
    });

    canvas.addEventListener("click", (event) => {
        if (!draft?.image || busy || draft.points.length >= 256) return;
        const rect = canvas.getBoundingClientRect();
        const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
        const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);
        draft.points.push([x, y]);
        draw();
        refreshControls();
    });

    undoBtn.addEventListener("click", () => {
        draft.points.pop();
        draw();
        refreshControls();
    });

    clearBtn.addEventListener("click", () => {
        draft.points = [];
        draw();
        refreshControls();
    });

    saveBtn.addEventListener("click", () => {
        void run(async () => {
            const active = draft;
            const { frame_width, frame_height } = active.frame;
            const response = await fetch(active.path, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame_width, frame_height, points: active.points }),
                signal: active.controller.signal,
            });
            if (!response.ok) throw new Error(await readBackendError(response));
            if (draft !== active) return;
            useCameraCalibrationInput.checked = true;
            setStatus(`Граница сохранена для камеры ${active.cameraId}`, "idle");
            dialog.close();
        }, "Сохраняется граница…");
    });

    refreshVideoOptions();
})();
