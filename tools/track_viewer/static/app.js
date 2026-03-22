const state = {
  folders: [],
  sync: true,
  showIds: true,
  lockEnabled: false,
  lockOffset: 0,
  cacheEnabled: false,
  scoreThreshold: 0,
  followSequence: true,
  diffOnly: false,
  diffIou: 0.5,
  diffFrames: [],
  diffScanBusy: false,
};

function qs(selector, root = document) {
  return root.querySelector(selector);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function colorForId(id) {
  const hue = (id * 47) % 360;
  return `hsl(${hue}, 70%, 45%)`;
}

async function fetchJson(url) {
  const res = await fetch(url);
  return res.json();
}

function buildPanel(panelEl) {
  const panel = {
    root: panelEl,
    side: panelEl.id === "rightPanel" ? "right" : "left",
    folderSelect: qs(".folderSelect", panelEl),
    sequenceSelect: qs(".sequenceSelect", panelEl),
    prevBtn: qs(".prevBtn", panelEl),
    nextBtn: qs(".nextBtn", panelEl),
    goBtn: qs(".goBtn", panelEl),
    frameInput: qs(".frameInput", panelEl),
    stepInput: qs(".stepInput", panelEl),
    rangeLabel: qs(".rangeLabel", panelEl),
    statusLabel: qs(".statusLabel", panelEl),
    countLabel: qs(".countLabel", panelEl),
    canvas: qs(".viewerCanvas", panelEl),
    emptyState: qs(".emptyState", panelEl),
    ctx: qs(".viewerCanvas", panelEl).getContext("2d"),
    minFrame: 1,
    maxFrame: 1,
    frame: 1,
    cacheFrames: null,
    lastBoxes: [],
  };

  panel.prevBtn.addEventListener("click", () => stepFrame(panel, -1));
  panel.nextBtn.addEventListener("click", () => stepFrame(panel, 1));
  panel.goBtn.addEventListener("click", () => jumpFrame(panel));
  panel.frameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      jumpFrame(panel);
    }
  });

  panel.folderSelect.addEventListener("change", () => {
    loadSequences(panel);
    state.diffFrames = [];
  });

  panel.sequenceSelect.addEventListener("change", () => {
    if (state.followSequence) {
      followSequence(panel);
    }
    loadFrameRange(panel, false);
    state.diffFrames = [];
  });

  panel.root.addEventListener("click", () => setActivePanel(panel));

  return panel;
}

async function loadFolders(panels) {
  const data = await fetchJson("/api/folders");
  state.folders = data.folders || [];
  panels.forEach((panel) => {
    panel.folderSelect.innerHTML = "";
    state.folders.forEach((folder) => {
      const option = document.createElement("option");
      option.value = folder;
      option.textContent = folder;
      panel.folderSelect.appendChild(option);
    });
  });

  panels.forEach((panel) => loadSequences(panel));
}

async function loadSequences(panel) {
  const folder = panel.folderSelect.value;
  if (!folder) {
    return;
  }
  const data = await fetchJson(`/api/sequences?folder=${encodeURIComponent(folder)}`);
  panel.sequenceSelect.innerHTML = "";
  (data.sequences || []).forEach((seq) => {
    const option = document.createElement("option");
    option.value = seq;
    option.textContent = seq;
    panel.sequenceSelect.appendChild(option);
  });

  loadFrameRange(panel, false);
}

async function loadFrameRange(panel, preserveFrame) {
  const folder = panel.folderSelect.value;
  const sequence = panel.sequenceSelect.value;
  if (!folder || !sequence) {
    return;
  }

  if (state.cacheEnabled) {
    await loadSequenceCache(panel, preserveFrame);
  } else {
    panel.cacheFrames = null;
    const data = await fetchJson(
      `/api/frames?folder=${encodeURIComponent(folder)}&sequence=${encodeURIComponent(sequence)}`
    );
    panel.minFrame = data.min || 1;
    panel.maxFrame = data.max || panel.minFrame;
    panel.frame = preserveFrame
      ? clamp(panel.frame, panel.minFrame, panel.maxFrame)
      : panel.minFrame;
    panel.frameInput.value = panel.frame;
    panel.rangeLabel.textContent = `帧范围: ${panel.minFrame} - ${panel.maxFrame}`;
    renderFrame(panel);
  }
  updateDeltaBadge();
}

async function loadSequenceCache(panel, preserveFrame) {
  const folder = panel.folderSelect.value;
  const sequence = panel.sequenceSelect.value;
  panel.statusLabel.textContent = "缓存中...";
  const data = await fetchJson(
    `/api/sequence?folder=${encodeURIComponent(folder)}&sequence=${encodeURIComponent(sequence)}`
  );
  panel.cacheFrames = data.frames || {};
  panel.minFrame = data.min || 1;
  panel.maxFrame = data.max || panel.minFrame;
  panel.frame = preserveFrame
    ? clamp(panel.frame, panel.minFrame, panel.maxFrame)
    : panel.minFrame;
  panel.frameInput.value = panel.frame;
  panel.rangeLabel.textContent = `帧范围: ${panel.minFrame} - ${panel.maxFrame}`;
  renderFrame(panel);
}

function stepFrame(panel, direction) {
  const step = Number(panel.stepInput.value) || 1;
  panel.frame = clamp(panel.frame + direction * step, panel.minFrame, panel.maxFrame);
  panel.frameInput.value = panel.frame;
  renderFrame(panel);
  syncIfNeeded(panel);
  updateDeltaBadge();
}

function jumpFrame(panel) {
  const target = Number(panel.frameInput.value) || panel.minFrame;
  panel.frame = clamp(target, panel.minFrame, panel.maxFrame);
  panel.frameInput.value = panel.frame;
  renderFrame(panel);
  syncIfNeeded(panel);
  updateDeltaBadge();
}

function syncIfNeeded(sourcePanel) {
  const panels = window.viewerPanels || [];
  if (state.lockEnabled) {
    panels.forEach((panel) => {
      if (panel === sourcePanel) {
        return;
      }
      const targetFrame =
        sourcePanel.side === "left"
          ? sourcePanel.frame + state.lockOffset
          : sourcePanel.frame - state.lockOffset;
      const clamped = clamp(targetFrame, panel.minFrame, panel.maxFrame);
      panel.frame = clamped;
      panel.frameInput.value = panel.frame;
      renderFrame(panel);
    });
    return;
  }

  if (!state.sync) {
    return;
  }
  panels.forEach((panel) => {
    if (panel === sourcePanel) {
      return;
    }
    const clamped = clamp(sourcePanel.frame, panel.minFrame, panel.maxFrame);
    panel.frame = clamped;
    panel.frameInput.value = panel.frame;
    renderFrame(panel);
  });
}

function followSequence(sourcePanel) {
  const panels = window.viewerPanels || [];
  const targetPanel = panels.find((panel) => panel !== sourcePanel);
  if (!targetPanel) {
    return;
  }
  const targetValue = sourcePanel.sequenceSelect.value;
  const options = Array.from(targetPanel.sequenceSelect.options).map(
    (option) => option.value
  );
  if (!options.includes(targetValue)) {
    return;
  }
  targetPanel.sequenceSelect.value = targetValue;
  loadFrameRange(targetPanel, true);
}

async function renderFrame(panel) {
  const folder = panel.folderSelect.value;
  const sequence = panel.sequenceSelect.value;
  const frame = panel.frame;
  if (!folder || !sequence) {
    return;
  }

  panel.statusLabel.textContent = "加载中...";
  let boxes = [];
  if (state.cacheEnabled && panel.cacheFrames) {
    boxes = panel.cacheFrames[String(frame)] || [];
  } else {
    const data = await fetchJson(
      `/api/frame?folder=${encodeURIComponent(folder)}&sequence=${encodeURIComponent(
        sequence
      )}&frame=${frame}`
    );
    panel.minFrame = data.min || panel.minFrame;
    panel.maxFrame = data.max || panel.maxFrame;
    panel.rangeLabel.textContent = `帧范围: ${panel.minFrame} - ${panel.maxFrame}`;
    boxes = data.boxes || [];
  }
  panel.lastBoxes = boxes;

  const imgUrl = `/api/image?folder=${encodeURIComponent(folder)}&sequence=${encodeURIComponent(
    sequence
  )}&frame=${frame}`;

  const image = new Image();
  image.onload = () => {
    panel.canvas.width = image.naturalWidth;
    panel.canvas.height = image.naturalHeight;
    panel.ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);
    panel.ctx.drawImage(image, 0, 0);
    drawBoxes(panel, boxes);
    panel.emptyState.style.display = "none";
    panel.statusLabel.textContent = `帧 ${frame}`;
    panel.countLabel.textContent = `框 ${countVisibleBoxes(panel, boxes)}`;
    updateCompareStats();
  };
  image.onerror = () => {
    panel.ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);
    panel.emptyState.style.display = "block";
    panel.statusLabel.textContent = "找不到图片";
  };
  image.src = imgUrl;
}

function drawBoxes(panel, boxes) {
  if (!boxes.length) {
    panel.countLabel.textContent = "框 0";
    return;
  }
  const threshold = state.scoreThreshold;
  const filtered = filterBoxes(panel, boxes, threshold);
  if (!filtered.length) {
    panel.countLabel.textContent = "框 0";
    return;
  }
  const compareInfo = getCompareInfo();
  const unmatched = getUnmatchedSet(panel, compareInfo);
  panel.ctx.lineWidth = Math.max(2, panel.canvas.width / 640);
  panel.ctx.font = `${Math.max(12, panel.canvas.width / 80)}px Space Grotesk`;
  panel.ctx.textBaseline = "top";

  filtered.forEach((box, index) => {
    const score = Number(box.score ?? 1);
    const color = colorForId(box.id || 0);
    const isUnmatched = unmatched ? unmatched.has(index) : false;
    panel.ctx.strokeStyle = color;
    panel.ctx.fillStyle = color;
    panel.ctx.setLineDash(isUnmatched ? [6, 4] : []);
    panel.ctx.lineWidth = isUnmatched
      ? Math.max(3, panel.canvas.width / 520)
      : Math.max(2, panel.canvas.width / 640);
    panel.ctx.strokeRect(box.x, box.y, box.w, box.h);

    const label = buildLabel(box, score);
    if (!label) {
      return;
    }
    const padding = 4;
    const metrics = panel.ctx.measureText(label);
    const boxHeight = Math.max(16, metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent + 4);
    panel.ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    panel.ctx.fillRect(box.x, box.y, metrics.width + padding * 2, boxHeight);
    panel.ctx.fillStyle = "#ffffff";
    panel.ctx.fillText(label, box.x + padding, box.y + 2);
  });
  panel.ctx.setLineDash([]);
}

function buildLabel(box, score) {
  const parts = [];
  if (state.showIds) {
    parts.push(`ID ${box.id}`);
  }
  parts.push(score.toFixed(2));
  return parts.join(" ");
}

function countVisibleBoxes(panel, boxes) {
  const threshold = state.scoreThreshold;
  return filterBoxes(panel, boxes, threshold).length;
}

function filterBoxes(panel, boxes, threshold) {
  let filtered = boxes.filter((box) => Number(box.score ?? 1) >= threshold);
  if (!state.diffOnly) {
    return filtered;
  }
  const compareInfo = getCompareInfo();
  const unmatched = getUnmatchedSet(panel, compareInfo);
  if (!unmatched) {
    return filtered;
  }
  return filtered.filter((_, index) => unmatched.has(index));
}

function getOtherPanel(panel) {
  const panels = window.viewerPanels || [];
  return panels.find((entry) => entry !== panel);
}

function iou(a, b) {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const interX1 = Math.max(a.x, b.x);
  const interY1 = Math.max(a.y, b.y);
  const interX2 = Math.min(ax2, bx2);
  const interY2 = Math.min(ay2, by2);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;
  if (interArea <= 0) {
    return 0;
  }
  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  return interArea / (areaA + areaB - interArea);
}

function getCompareInfo() {
  const panels = window.viewerPanels || [];
  const leftPanel = panels.find((panel) => panel.side === "left");
  const rightPanel = panels.find((panel) => panel.side === "right");
  if (!leftPanel || !rightPanel) {
    return null;
  }
  if (leftPanel.frame !== rightPanel.frame) {
    return null;
  }
  const threshold = state.scoreThreshold;
  const leftFiltered = leftPanel.lastBoxes.filter(
    (box) => Number(box.score ?? 1) >= threshold
  );
  const rightFiltered = rightPanel.lastBoxes.filter(
    (box) => Number(box.score ?? 1) >= threshold
  );
  const leftOnly = new Set();
  const rightOnly = new Set();
  const matchedRight = new Set();

  leftFiltered.forEach((leftBox, leftIndex) => {
    let matched = false;
    rightFiltered.forEach((rightBox, rightIndex) => {
      if (leftBox.id === rightBox.id && iou(leftBox, rightBox) >= state.diffIou) {
        matched = true;
        matchedRight.add(rightIndex);
      }
    });
    if (!matched) {
      leftOnly.add(leftIndex);
    }
  });

  rightFiltered.forEach((_, rightIndex) => {
    if (!matchedRight.has(rightIndex)) {
      rightOnly.add(rightIndex);
    }
  });

  const matchedCount = Math.min(
    leftFiltered.length - leftOnly.size,
    rightFiltered.length - rightOnly.size
  );

  return {
    frame: leftPanel.frame,
    leftOnly,
    rightOnly,
    matchedCount,
    leftCount: leftFiltered.length,
    rightCount: rightFiltered.length,
  };
}

async function ensureFrameMaps() {
  const panels = window.viewerPanels || [];
  const leftPanel = panels.find((panel) => panel.side === "left");
  const rightPanel = panels.find((panel) => panel.side === "right");
  if (!leftPanel || !rightPanel) {
    return null;
  }

  const leftMap = await getFrameMap(leftPanel);
  const rightMap = await getFrameMap(rightPanel);
  return {
    leftPanel,
    rightPanel,
    leftMap,
    rightMap,
  };
}

async function getFrameMap(panel) {
  if (panel.cacheFrames) {
    return panel.cacheFrames;
  }
  const folder = panel.folderSelect.value;
  const sequence = panel.sequenceSelect.value;
  if (!folder || !sequence) {
    return {};
  }
  const data = await fetchJson(
    `/api/sequence?folder=${encodeURIComponent(folder)}&sequence=${encodeURIComponent(sequence)}`
  );
  panel.cacheFrames = data.frames || {};
  panel.minFrame = data.min || panel.minFrame;
  panel.maxFrame = data.max || panel.maxFrame;
  panel.rangeLabel.textContent = `帧范围: ${panel.minFrame} - ${panel.maxFrame}`;
  return panel.cacheFrames;
}

async function scanDiffFrames() {
  if (state.diffScanBusy) {
    return;
  }
  state.diffScanBusy = true;
  const stats = qs("#matchStats");
  stats.textContent = "扫描中...";

  const maps = await ensureFrameMaps();
  if (!maps) {
    state.diffScanBusy = false;
    updateCompareStats();
    return;
  }
  const { leftPanel, rightPanel, leftMap, rightMap } = maps;
  const start = Math.max(leftPanel.minFrame, rightPanel.minFrame);
  const end = Math.min(leftPanel.maxFrame, rightPanel.maxFrame);
  const diffFrames = [];
  const threshold = state.scoreThreshold;

  for (let frame = start; frame <= end; frame += 1) {
    const leftBoxes = (leftMap[String(frame)] || []).filter(
      (box) => Number(box.score ?? 1) >= threshold
    );
    const rightBoxes = (rightMap[String(frame)] || []).filter(
      (box) => Number(box.score ?? 1) >= threshold
    );
    const diffInfo = compareBoxes(leftBoxes, rightBoxes);
    if (diffInfo.leftOnly.size > 0 || diffInfo.rightOnly.size > 0) {
      diffFrames.push(frame);
    }
  }

  state.diffFrames = diffFrames;
  state.diffScanBusy = false;
  stats.textContent = diffFrames.length
    ? `差异帧: ${diffFrames.length}`
    : "未发现差异帧";
}

function compareBoxes(leftBoxes, rightBoxes) {
  const leftOnly = new Set();
  const rightOnly = new Set();
  const matchedRight = new Set();

  leftBoxes.forEach((leftBox, leftIndex) => {
    let matched = false;
    rightBoxes.forEach((rightBox, rightIndex) => {
      if (leftBox.id === rightBox.id && iou(leftBox, rightBox) >= state.diffIou) {
        matched = true;
        matchedRight.add(rightIndex);
      }
    });
    if (!matched) {
      leftOnly.add(leftIndex);
    }
  });

  rightBoxes.forEach((_, rightIndex) => {
    if (!matchedRight.has(rightIndex)) {
      rightOnly.add(rightIndex);
    }
  });

  return { leftOnly, rightOnly };
}

async function jumpToDiff(direction) {
  if (!state.diffFrames.length) {
    await scanDiffFrames();
  }
  if (!state.diffFrames.length) {
    return;
  }
  const panels = window.viewerPanels || [];
  const leftPanel = panels.find((panel) => panel.side === "left");
  const rightPanel = panels.find((panel) => panel.side === "right");
  if (!leftPanel || !rightPanel) {
    return;
  }
  const current = leftPanel.frame;
  const sorted = state.diffFrames.slice().sort((a, b) => a - b);
  let target = current;
  if (direction > 0) {
    target = sorted.find((frame) => frame > current) ?? sorted[0];
  } else {
    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      if (sorted[i] < current) {
        target = sorted[i];
        break;
      }
    }
    if (target === current) {
      target = sorted[sorted.length - 1];
    }
  }

  leftPanel.frame = clamp(target, leftPanel.minFrame, leftPanel.maxFrame);
  leftPanel.frameInput.value = leftPanel.frame;
  renderFrame(leftPanel);
  syncIfNeeded(leftPanel);
  updateDeltaBadge();
}

function getUnmatchedSet(panel, compareInfo) {
  if (!compareInfo) {
    return null;
  }
  return panel.side === "left" ? compareInfo.leftOnly : compareInfo.rightOnly;
}

function updateCompareStats() {
  const stats = qs("#matchStats");
  const compareInfo = getCompareInfo();
  if (!compareInfo) {
    stats.textContent = "匹配: -- | 左独有: -- | 右独有: --";
    return;
  }
  stats.textContent = `匹配: ${compareInfo.matchedCount} | 左独有: ${compareInfo.leftOnly.size} | 右独有: ${compareInfo.rightOnly.size}`;
}

function setActivePanel(panel) {
  const panels = window.viewerPanels || [];
  panels.forEach((entry) => entry.root.classList.toggle("active", entry === panel));
  window.activePanel = panel;
}

function updateDeltaBadge() {
  const deltaBadge = qs("#deltaBadge");
  const panels = window.viewerPanels || [];
  if (panels.length < 2) {
    deltaBadge.textContent = "Δ: 0";
    return;
  }
  const leftPanel = panels.find((panel) => panel.side === "left");
  const rightPanel = panels.find((panel) => panel.side === "right");
  if (!leftPanel || !rightPanel) {
    deltaBadge.textContent = "Δ: 0";
    return;
  }
  const delta = rightPanel.frame - leftPanel.frame;
  const prefix = delta > 0 ? "+" : "";
  deltaBadge.textContent = `Δ: ${prefix}${delta}`;
}

function init() {
  const panels = [buildPanel(qs("#leftPanel")), buildPanel(qs("#rightPanel"))];
  window.viewerPanels = panels;

  qs("#syncToggle").addEventListener("change", (event) => {
    state.sync = event.target.checked;
  });

  qs("#lockToggle").addEventListener("change", (event) => {
    state.lockEnabled = event.target.checked;
    const leftPanel = panels.find((panel) => panel.side === "left");
    const rightPanel = panels.find((panel) => panel.side === "right");
    if (leftPanel && rightPanel) {
      state.lockOffset = rightPanel.frame - leftPanel.frame;
    }
    updateDeltaBadge();
  });

  qs("#idToggle").addEventListener("change", (event) => {
    state.showIds = event.target.checked;
    panels.forEach((panel) => renderFrame(panel));
  });

  qs("#cacheToggle").addEventListener("change", (event) => {
    state.cacheEnabled = event.target.checked;
    panels.forEach((panel) => loadFrameRange(panel, true));
    updateCompareStats();
  });

  qs("#scoreThreshold").addEventListener("change", (event) => {
    state.scoreThreshold = Number(event.target.value) || 0;
    document.querySelectorAll(".threshold-buttons .mini").forEach((btn) => {
      const value = Number(btn.dataset.threshold);
      btn.classList.toggle("active", value === state.scoreThreshold);
    });
    panels.forEach((panel) => renderFrame(panel));
    state.diffFrames = [];
  });

  qs("#iouThreshold").addEventListener("change", (event) => {
    state.diffIou = Number(event.target.value) || 0.5;
    panels.forEach((panel) => renderFrame(panel));
    state.diffFrames = [];
  });

  qs("#followToggle").addEventListener("change", (event) => {
    state.followSequence = event.target.checked;
  });

  qs("#diffToggle").addEventListener("change", (event) => {
    state.diffOnly = event.target.checked;
    panels.forEach((panel) => renderFrame(panel));
    updateCompareStats();
    state.diffFrames = [];
  });

  document.querySelectorAll(".threshold-buttons .mini").forEach((button) => {
    button.addEventListener("click", () => {
      const value = Number(button.dataset.threshold);
      qs("#scoreThreshold").value = value.toFixed(2);
      state.scoreThreshold = value;
      document.querySelectorAll(".threshold-buttons .mini").forEach((btn) => {
        btn.classList.toggle("active", btn === button);
      });
      panels.forEach((panel) => renderFrame(panel));
      state.diffFrames = [];
    });
  });

  qs("#scanDiffBtn").addEventListener("click", () => {
    scanDiffFrames();
  });

  qs("#prevDiffBtn").addEventListener("click", () => {
    jumpToDiff(-1);
  });

  qs("#nextDiffBtn").addEventListener("click", () => {
    jumpToDiff(1);
  });

  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) {
      return;
    }
    const activePanel = window.activePanel || panels[0];
    if (event.key === "ArrowLeft") {
      stepFrame(activePanel, -1);
    }
    if (event.key === "ArrowRight") {
      stepFrame(activePanel, 1);
    }
  });

  loadFolders(panels);
  setActivePanel(panels[0]);
}

window.addEventListener("DOMContentLoaded", init);
