// ==========================================================================
// Multimodal Industrial Anomaly Diagnostics UI - Client Logic
// Connected to FastAPI Backend (PyTorch RGB + Depth + GACM + PatchCore Models)
// ==========================================================================

const API_BASE_URL = ""; // Relative URL points to current FastAPI origin

const state = {
    currentFile: null,
    currentBase64: null,
    isScanning: false,
    webcamStream: null
};

// DOM Elements
const categorySelect = document.getElementById('categorySelect');
const systemStatusPill = document.getElementById('systemStatusPill');
const systemStatusText = document.getElementById('systemStatusText');

const dropZone = document.getElementById('dropZone');
const emptyState = document.getElementById('emptyState');
const activeImageView = document.getElementById('activeImageView');
const previewImage = document.getElementById('previewImage');
const radarScanOverlay = document.getElementById('radarScanOverlay');
const regionHighlightBox = document.getElementById('regionHighlightBox');
const fileInput = document.getElementById('fileInput');

const btnOpenWebcam = document.getElementById('btnOpenWebcam');
const webcamOverlay = document.getElementById('webcamOverlay');
const webcamFeed = document.getElementById('webcamFeed');
const btnCaptureFrame = document.getElementById('btnCaptureFrame');
const btnCloseWebcam = document.getElementById('btnCloseWebcam');
const captureCanvas = document.getElementById('captureCanvas');

const textPrompt = document.getElementById('textPrompt');
const btnRunInspect = document.getElementById('btnRunInspect');
const btnInspectText = document.getElementById('btnInspectText');
const btnReset = document.getElementById('btnReset');
const samplePills = document.querySelectorAll('.pill-chip');

const diagEmptyState = document.getElementById('diagEmptyState');
const diagContent = document.getElementById('diagContent');
const verdictCard = document.getElementById('verdictCard');
const verdictTitle = document.getElementById('verdictTitle');
const verdictDesc = document.getElementById('verdictDesc');
const confidenceScore = document.getElementById('confidenceScore');
const severityTag = document.getElementById('severityTag');
const tableBody = document.getElementById('tableBody');
const analysisHeader = document.getElementById('analysisHeader');
const analysisCardsContainer = document.getElementById('analysisCardsContainer');
const specsChips = document.getElementById('specsChips');
const scanTimestamp = document.getElementById('scanTimestamp');

// ── Sample Presets ────────────────────────────────────────────────────────
const SAMPLE_PRESETS = {
    normal_phone: {
        category: "phone_screen",
        prompt: "flawless clean screen",
        url: "/static/presets/normal_phone.png"
    },
    cracked_phone: {
        category: "phone_screen",
        prompt: "top-right cracked screen corner",
        url: "/static/presets/cracked_phone.png"
    },
    broken_phone: {
        category: "phone_screen",
        prompt: "broken shattered screen",
        url: "/static/presets/broken_phone.png"
    }
};

// ── Initialization ────────────────────────────────────────────────────────
function init() {
    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-active');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-active');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Webcam
    btnOpenWebcam.addEventListener('click', openWebcam);
    btnCloseWebcam.addEventListener('click', closeWebcam);
    btnCaptureFrame.addEventListener('click', captureWebcamFrame);

    // Prompt Enter Key
    textPrompt.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            runInspection();
        }
    });

    // Run Inspect Button
    btnRunInspect.addEventListener('click', runInspection);

    // Reset
    btnReset.addEventListener('click', resetView);

    // Sample Preset Buttons
    samplePills.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleKey = btn.getAttribute('data-sample');
            loadSamplePreset(sampleKey);
        });
    });
}

// ── File & Image Handling ────────────────────────────────────────────────
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file (PNG, JPG, JPEG, WEBP).');
        return;
    }

    state.currentFile = file;
    state.currentBase64 = null;

    const reader = new FileReader();
    reader.onload = (e) => {
        showPreviewImage(e.target.result);
    };
    reader.readAsDataURL(file);
}

function showPreviewImage(src) {
    previewImage.src = src;
    emptyState.style.display = 'none';
    webcamOverlay.style.display = 'none';
    activeImageView.style.display = 'flex';
    regionHighlightBox.style.display = 'none';
}

function resetView() {
    state.currentFile = null;
    state.currentBase64 = null;
    state.isScanning = false;

    closeWebcam();
    fileInput.value = '';
    textPrompt.value = '';
    previewImage.src = '';

    activeImageView.style.display = 'none';
    emptyState.style.display = 'flex';
    radarScanOverlay.classList.remove('scanning');
    regionHighlightBox.style.display = 'none';

    diagContent.style.display = 'none';
    diagEmptyState.style.display = 'flex';
    scanTimestamp.textContent = 'Ready';
}

// ── Webcam Handler ────────────────────────────────────────────────────────
async function openWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });
        state.webcamStream = stream;
        webcamFeed.srcObject = stream;
        emptyState.style.display = 'none';
        activeImageView.style.display = 'none';
        webcamOverlay.style.display = 'flex';
    } catch (err) {
        alert('Camera access denied or unavailable: ' + err.message);
    }
}

function closeWebcam() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        state.webcamStream = null;
    }
    webcamOverlay.style.display = 'none';
    if (!state.currentFile && !state.currentBase64) {
        emptyState.style.display = 'flex';
    }
}

function captureWebcamFrame() {
    if (!state.webcamStream) return;

    captureCanvas.width = webcamFeed.videoWidth || 640;
    captureCanvas.height = webcamFeed.videoHeight || 480;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(webcamFeed, 0, 0, captureCanvas.width, captureCanvas.height);

    const base64Data = captureCanvas.toDataURL('image/jpeg', 0.92);
    state.currentBase64 = base64Data;
    state.currentFile = null;

    closeWebcam();
    showPreviewImage(base64Data);
}

// ── Sample Presets Loader ────────────────────────────────────────────────
function loadSamplePreset(presetKey) {
    const p = SAMPLE_PRESETS[presetKey];
    if (!p) return;

    categorySelect.value = p.category;
    textPrompt.value = p.prompt;

    fetch(p.url)
        .then(res => {
            if (!res.ok) throw new Error('Preset image missing');
            return res.blob();
        })
        .then(blob => {
            const file = new File([blob], presetKey + '.png', { type: 'image/png' });
            handleFileSelect(file);
            // Automatically trigger inspection on sample load
            setTimeout(() => runInspection(), 250);
        })
        .catch(err => {
            console.error('Failed to load preset:', err);
        });
}

// ── Run Multimodal Inspection ────────────────────────────────────────────
async function runInspection() {
    if (!state.currentFile && !state.currentBase64) {
        alert('Please upload or capture an image before running diagnostics. Image input is compulsory.');
        return;
    }

    if (state.isScanning) return;
    state.isScanning = true;

    // UI Loading state
    btnRunInspect.classList.add('loading');
    btnInspectText.textContent = 'Analyzing...';
    radarScanOverlay.classList.add('scanning');
    systemStatusText.textContent = 'Inference in Progress...';
    systemStatusPill.style.background = 'rgba(99, 102, 241, 0.15)';
    systemStatusPill.style.color = '#818CF8';

    const formData = new FormData();
    if (state.currentFile) {
        formData.append('file', state.currentFile);
    } else if (state.currentBase64) {
        formData.append('image_base64', state.currentBase64);
    }

    const promptVal = textPrompt.value.trim() || 'flawless';
    formData.append('prompt', promptVal);
    formData.append('category', categorySelect.value);

    try {
        const response = await fetch(API_BASE_URL + '/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Server returned HTTP ' + response.status);
        }

        const data = await response.json();
        renderDiagnosticsResults(data);
    } catch (err) {
        console.error('Inspection failed:', err);
        alert('Inspection request failed. Make sure the backend server is running.\n' + err.message);
    } finally {
        state.isScanning = false;
        btnRunInspect.classList.remove('loading');
        btnInspectText.textContent = 'Run Diagnostics';
        radarScanOverlay.classList.remove('scanning');
        systemStatusText.textContent = 'Model Ready';
        systemStatusPill.style.background = 'rgba(16, 185, 129, 0.1)';
        systemStatusPill.style.color = 'var(--status-success)';
    }
}

// ── Render Structured Telemetry Output ────────────────────────────────────
function renderDiagnosticsResults(data) {
    diagEmptyState.style.display = 'none';
    diagContent.style.display = 'flex';
    scanTimestamp.textContent = new Date().toLocaleTimeString();

    // 1. Verdict Hero Card
    const isAnomaly = data.is_anomaly;
    verdictCard.className = 'verdict-card ' + (isAnomaly ? 'critical' : 'success');
    verdictTitle.textContent = isAnomaly ? 'ANOMALY DETECTED' : 'FLAWLESS / NORMAL';
    verdictDesc.textContent = isAnomaly
        ? 'Elevated spatial patch divergence detected on active ' + (data.category || '').replace('_', ' ') + ' surface.'
        : 'All spatial PatchCore features match normal baseline distribution with zero critical defects.';

    confidenceScore.textContent = data.confidence || '95.0%';
    severityTag.textContent = 'Severity: ' + (data.severity || (isAnomaly ? 'High' : 'Low'));

    // 2. Telemetry Table (Exact Requested Format)
    tableBody.innerHTML = '';
    const rows = data.diagnostics_table || [];

    rows.forEach(r => {
        const tr = document.createElement('tr');

        // Metric cell
        const tdMetric = document.createElement('td');
        tdMetric.className = 'metric-name-cell';
        tdMetric.textContent = r.metric;

        // Measurement cell
        const tdMeas = document.createElement('td');
        tdMeas.className = 'metric-meas-cell';
        tdMeas.textContent = r.measurement;

        // Threshold cell
        const tdThresh = document.createElement('td');
        tdThresh.className = 'metric-thresh-cell';
        tdThresh.textContent = r.threshold;

        // Evaluation cell
        const tdEval = document.createElement('td');
        const badge = document.createElement('span');
        badge.className = 'eval-badge ' + (r.status || 'ok');
        badge.textContent = r.evaluation;
        tdEval.appendChild(badge);

        tr.appendChild(tdMetric);
        tr.appendChild(tdMeas);
        tr.appendChild(tdThresh);
        tr.appendChild(tdEval);
        tableBody.appendChild(tr);
    });

    // 3. Detailed Inspection & Resolution Analysis Section
    analysisCardsContainer.innerHTML = '';
    const inspectionData = data.inspection_analysis;

    if (inspectionData && inspectionData.sections) {
        inspectionData.sections.forEach(sec => {
            const card = document.createElement('div');
            card.className = 'analysis-card';

            const header = document.createElement('div');
            header.className = 'analysis-card-header';
            header.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg><span>' + sec.heading + '</span>';

            const ul = document.createElement('ul');
            ul.className = 'analysis-bullets-list';
            sec.bullets.forEach(bulletText => {
                const li = document.createElement('li');
                li.textContent = bulletText;
                ul.appendChild(li);
            });

            card.appendChild(header);
            card.appendChild(ul);
            analysisCardsContainer.appendChild(card);
        });
    }

    // 4. Specs Chips
    specsChips.innerHTML = '';
    const specs = data.specs || [];
    specs.forEach(s => {
        const chip = document.createElement('div');
        chip.className = 'spec-chip';
        chip.innerHTML = '<strong>' + s.name + ':</strong> ' + s.val;
        specsChips.appendChild(chip);
    });
}

// Start
document.addEventListener('DOMContentLoaded', init);
