(() => {
    const dialog = document.getElementById("calibrationDialog");
    const title = document.getElementById("calibrationTitle");
    const canvas = document.getElementById("calibrationCanvas");
    const ctx = canvas.getContext("2d");
    const viewport = document.getElementById("calibrationViewport");
    const zoomControls = document.getElementById("calibrationZoomControls");
    const zoomLevel = document.getElementById("calibrationZoomLevel");
    const zoomInBtn = document.getElementById("calibrationZoomIn");
    const zoomOutBtn = document.getElementById("calibrationZoomOut");
    const zoomResetBtn = document.getElementById("calibrationZoomReset");
    const status = document.getElementById("calibrationStatus");
    const time = document.getElementById("calibrationTime");
    const frameTime = document.getElementById("calibrationFrameTime");
    const previewForm = document.getElementById("calibrationPreviewForm");
    const previewBtn = document.getElementById("calibrationPreviewBtn");
    const closeBtn = document.getElementById("closeCalibrationBtn");
    const saveBtn = document.getElementById("saveCalibrationBtn");
    const undoBtn = document.getElementById("undoCalibrationPoint");
    const clearBtn = document.getElementById("clearCalibrationPoints");

    const tools = document.getElementById("calibrationTools");
    const toolButtons = [...tools.querySelectorAll("[data-calibration-tool]")];
    const brushControls = document.getElementById("calibrationBrushControls");
    const brushSize = document.getElementById("calibrationBrushSize");
    const brushValue = document.getElementById("calibrationBrushValue");
    const maskCanvas = document.createElement("canvas");
    const maskCtx = maskCanvas.getContext("2d");

    let draft = null;
    let gesture = null;
    let brushPosition = null;
    let renderPending = false;
    let busy = false;
    let zoom = 1;

    function setZoom(value) {
        const x = (viewport.scrollLeft + viewport.clientWidth / 2) / Math.max(1, canvas.offsetWidth);
        const y = (viewport.scrollTop + viewport.clientHeight / 2) / Math.max(1, canvas.offsetHeight);
        zoom = clamp(value, 1, 4);
        canvas.style.width = `${zoom * 100}%`;
        zoomLevel.textContent = `${zoom * 100}%`;
        viewport.scrollLeft = x * canvas.offsetWidth - viewport.clientWidth / 2;
        viewport.scrollTop = y * canvas.offsetHeight - viewport.clientHeight / 2;
        zoomOutBtn.disabled = zoomResetBtn.disabled = zoom === 1;
        zoomInBtn.disabled = zoom === 4;
        draw();
    }

    zoomInBtn.addEventListener("click", () => setZoom(zoom + 0.5));
    zoomOutBtn.addEventListener("click", () => setZoom(zoom - 0.5));
    zoomResetBtn.addEventListener("click", () => setZoom(1));

    function refreshControls() {
        const editing = busy || gesture !== null;
        previewBtn.disabled = time.disabled = editing;
        undoBtn.disabled = editing || !draft?.history.length;
        clearBtn.disabled = editing || !(draft?.points.length || draft?.strokes.length);
        saveBtn.disabled = editing || !draft?.image
            || !(draft.points.length >= 3 || draft.strokes.some(stroke => stroke.tool === "paint"));
        for (const button of toolButtons) {
            button.disabled = editing || !draft?.image;
            button.setAttribute("aria-pressed", String(button.dataset.calibrationTool === draft?.tool));
        }
        brushControls.hidden = !draft || draft.tool === "points";
        brushSize.disabled = editing;
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
        gesture = null;
        brushPosition = null;
        canvas.removeAttribute("data-point-hover");
        canvas.removeAttribute("data-dragging");
        busy = false;
    });

    function tracePoints(context, points) {
        const { width, height } = canvas;
        context.beginPath();
        context.moveTo(points[0][0] * width, points[0][1] * height);
        for (const [x, y] of points.slice(1)) context.lineTo(x * width, y * height);
    }

    function drawBrushStroke(stroke) {
        const { width, height } = canvas;
        maskCtx.globalCompositeOperation = stroke.tool === "erase" ? "destination-out" : "source-over";
        maskCtx.lineWidth = stroke.size;
        maskCtx.lineCap = "round";
        maskCtx.lineJoin = "round";
        if (stroke.points.length > 1) {
            tracePoints(maskCtx, stroke.points);
            maskCtx.stroke();
        } else {
            const [[x, y]] = stroke.points;
            maskCtx.beginPath();
            maskCtx.arc(x * width, y * height, stroke.size / 2, 0, 2 * Math.PI);
            maskCtx.fill();
        }
    }

    function renderMask() {
        const { width, height } = canvas;
        maskCtx.clearRect(0, 0, width, height);
        maskCtx.fillStyle = maskCtx.strokeStyle = "#00cf9e";
        if (draft.points.length >= 3) {
            tracePoints(maskCtx, draft.points);
            maskCtx.closePath();
            maskCtx.fill();
        }
        for (const stroke of draft.strokes) drawBrushStroke(stroke);
        if (gesture?.stroke) drawBrushStroke(gesture.stroke);
        maskCtx.globalCompositeOperation = "source-over";
    }

    function draw() {
        if (!draft?.image) return;
        const { points, image, tool } = draft;
        const { width, height } = canvas;
        const pixelScale = width / Math.max(1, canvas.clientWidth);
        ctx.drawImage(image, 0, 0, width, height);
        renderMask();
        ctx.globalAlpha = 0.26;
        ctx.drawImage(maskCanvas, 0, 0);
        ctx.globalAlpha = 1;

        if (tool === "points" && points.length) {
            tracePoints(ctx, points);
            if (points.length >= 3) ctx.closePath();
            ctx.strokeStyle = "#60ffd5";
            ctx.lineWidth = 2 * pixelScale;
            ctx.stroke();
            points.forEach(([x, y], index) => {
                ctx.beginPath();
                ctx.arc(x * width, y * height, 4.5 * pixelScale, 0, 2 * Math.PI);
                ctx.fillStyle = index === gesture?.index ? "#ffcc66" : "#ffffff";
                ctx.fill();
            });
        }
        if (tool !== "points" && brushPosition) {
            const [x, y] = brushPosition;
            ctx.beginPath();
            ctx.arc(x * width, y * height, (gesture?.stroke?.size ?? Number(brushSize.value)) / 2, 0, 2 * Math.PI);
            ctx.lineWidth = 3 * pixelScale;
            ctx.strokeStyle = "#101624";
            ctx.stroke();
            ctx.lineWidth = pixelScale;
            ctx.strokeStyle = "#ffffff";
            ctx.stroke();
        }
    }

    function scheduleDraw() {
        if (renderPending) return;
        renderPending = true;
        requestAnimationFrame(() => {
            renderPending = false;
            draw();
        });
    }

    function snapshot() {
        return { points: draft.points, strokes: draft.strokes, tool: draft.tool };
    }

    function remember(previous = snapshot()) {
        draft.history.push(previous);
        if (draft.history.length > 50) draft.history.shift();
    }

    function pointAt(event) {
        const rect = canvas.getBoundingClientRect();
        return [
            clamp((event.clientX - rect.left) / rect.width, 0, 1),
            clamp((event.clientY - rect.top) / rect.height, 0, 1),
        ];
    }

    function nearbyPoint(event) {
        const rect = canvas.getBoundingClientRect();
        let closest = -1;
        let distance = 12;
        draft.points.forEach(([x, y], index) => {
            const current = Math.hypot(rect.left + x * rect.width - event.clientX,
                                       rect.top + y * rect.height - event.clientY);
            if (current < distance) {
                distance = current;
                closest = index;
            }
        });
        return closest;
    }

    async function applyBrushEdits() {
        const active = draft;
        if (!active.strokes.length) return;
        renderMask();
        const png = await new Promise(resolve => maskCanvas.toBlob(resolve, "image/png"));
        if (active !== draft) return;
        if (!png) throw new Error("Не удалось подготовить закрашенную область.");
        const request = new FormData();
        request.append("mask", png, "track-mask.png");
        const response = await fetch("/api/calibration/outline", {
            method: "POST",
            body: request,
            signal: active.controller.signal,
        });
        if (!response.ok) throw new Error(await readBackendError(response));
        const outline = await response.json();
        if (active !== draft) return;
        active.points = outline.points;
        active.strokes = [];
        draw();
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
        if (incompatible) {
            active.points = [];
            active.strokes = [];
            active.history = [];
        }
        draft.frame = frame;
        draft.image = image;
        canvas.width = maskCanvas.width = image.naturalWidth;
        canvas.height = maskCanvas.height = image.naturalHeight;
        viewport.hidden = false;
        zoomControls.hidden = false;
        tools.hidden = false;
        if (firstPreview) setZoom(1);
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
            controller: new AbortController(), tool: "points",
            points: [], strokes: [], history: [], frame: null, image: null,
        };
        title.textContent = `Граница трассы · ${cameraId}`;
        viewport.hidden = true;
        zoomControls.hidden = true;
        tools.hidden = true;
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
        if (!busy && !gesture) void run(loadPreview, "Загружается кадр…");
    });

    for (const button of toolButtons) {
        button.addEventListener("click", () => {
            if (!draft?.image || busy || gesture) return;
            const tool = button.dataset.calibrationTool;
            const select = () => {
                draft.tool = tool;
                brushPosition = null;
                canvas.removeAttribute("data-point-hover");
                refreshControls();
                draw();
            };
            if (tool === "points" && draft.strokes.length) {
                const active = draft;
                void run(async () => {
                    await applyBrushEdits();
                    if (active === draft) select();
                }, "Строится контур…");
            } else {
                select();
            }
        });
    }

    brushSize.addEventListener("input", () => {
        brushValue.textContent = brushSize.value + " px";
        draw();
    });

    canvas.addEventListener("pointerdown", event => {
        if (!draft?.image || busy || gesture || event.button !== 0 || !event.isPrimary) return;
        const point = pointAt(event);
        if (draft.tool === "points") {
            const index = nearbyPoint(event);
            if (index < 0) {
                if (draft.points.length >= 256) return;
                remember();
                draft.points = [...draft.points, point];
                draw();
                refreshControls();
                return;
            }
            const [x, y] = draft.points[index];
            gesture = { id: event.pointerId, index, before: snapshot(), offset: [x - point[0], y - point[1]] };
            canvas.setAttribute("data-dragging", "");
        } else {
            gesture = {
                id: event.pointerId,
                stroke: { tool: draft.tool, size: Number(brushSize.value), points: [point] },
            };
            brushPosition = point;
        }
        canvas.setPointerCapture(event.pointerId);
        draw();
        refreshControls();
    });

    canvas.addEventListener("pointermove", event => {
        if (!draft?.image || busy || !event.isPrimary || (gesture && gesture.id !== event.pointerId)) return;
        if (draft.tool === "points" && !gesture) {
            canvas.toggleAttribute("data-point-hover", nearbyPoint(event) >= 0);
            return;
        }
        const point = pointAt(event);
        if (gesture?.index !== undefined) {
            const moved = point.map((value, axis) => clamp(value + gesture.offset[axis], 0, 1));
            const current = draft.points[gesture.index];
            if (current[0] !== moved[0] || current[1] !== moved[1]) {
                draft.points = draft.points.map((value, index) => index === gesture.index ? moved : value);
            }
        } else {
            brushPosition = point;
            if (gesture?.stroke) {
                const points = gesture.stroke.points;
                const [x, y] = points[points.length - 1];
                if (Math.hypot((point[0] - x) * canvas.width, (point[1] - y) * canvas.height) >= 1) {
                    points.push(point);
                }
            }
        }
        scheduleDraw();
    });

    function finishGesture(event, cancelled = false) {
        if (!gesture || gesture.id !== event.pointerId || !draft) return;
        const active = gesture;
        gesture = null;
        if (active.stroke && !cancelled) {
            remember();
            draft.strokes = [...draft.strokes, active.stroke];
        }
        if (active.before) {
            if (cancelled) draft.points = active.before.points;
            else if (draft.points !== active.before.points) remember(active.before);
        }
        canvas.removeAttribute("data-dragging");
        draw();
        refreshControls();
    }

    canvas.addEventListener("pointerup", event => finishGesture(event));
    canvas.addEventListener("pointercancel", event => finishGesture(event, true));
    canvas.addEventListener("lostpointercapture", event => finishGesture(event, true));
    canvas.addEventListener("pointerleave", () => {
        brushPosition = null;
        canvas.removeAttribute("data-point-hover");
        scheduleDraw();
    });

    undoBtn.addEventListener("click", () => {
        const previous = draft?.history.pop();
        if (!previous) return;
        Object.assign(draft, previous);
        brushPosition = null;
        draw();
        refreshControls();
    });

    clearBtn.addEventListener("click", () => {
        remember();
        draft.points = [];
        draft.strokes = [];
        draw();
        refreshControls();
    });

    saveBtn.addEventListener("click", () => {
        void run(async () => {
            const active = draft;
            await applyBrushEdits();
            if (active !== draft) return;
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
