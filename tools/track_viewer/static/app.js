const state = {
  folders: [],
  sync: true,
  showIds: true,
  showGtIds: false,
  showScore: true,
  lockEnabled: false,
  lockOffset: 0,
  cacheEnabled: false,
  scoreThreshold: 0,
  followSequence: true,
  diffOnly: false,
  diffIou: 0.5,
  diffFrames: [],
  diffScanBusy: false,
  previewOpen: false,
  previewFilename: "",
  previewCanvas: null,
};

let previewDom = null;
const GT_UNMATCHED_KEY = "__unmatched__";

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

function sanitizeFilenamePart(value) {
  return String(value || "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "track_viewer";
}

function cloneCanvas(sourceCanvas) {
  const snapshot = document.createElement("canvas");
  snapshot.width = sourceCanvas.width;
  snapshot.height = sourceCanvas.height;
  snapshot.getContext("2d").drawImage(sourceCanvas, 0, 0);
  return snapshot;
}

function buildPreviewFilename(panel) {
  const folder = sanitizeFilenamePart(panel.folderSelect.value);
  const sequence = sanitizeFilenamePart(panel.sequenceSelect.value);
  const frame = String(panel.frame).padStart(6, "0");
  return `${folder}_${sequence}_${panel.side}_frame_${frame}.png`;
}

function openPreviewModal(panel) {
  if (
    !previewDom ||
    panel.isLoading ||
    !panel.lastImage ||
    !panel.canvas.width ||
    !panel.canvas.height
  ) {
    return;
  }
  const snapshot = cloneCanvas(panel.canvas);
  state.previewCanvas = snapshot;
  state.previewFilename = buildPreviewFilename(panel);
  state.previewOpen = true;
  previewDom.title.textContent = `${panel.side === "left" ? "左侧" : "右侧"} | ${
    panel.sequenceSelect.value || "--"
  } | 帧 ${panel.frame}`;
  previewDom.image.src = snapshot.toDataURL("image/png");
  previewDom.root.hidden = false;
  document.body.classList.add("modal-open");
}

function closePreviewModal() {
  if (!previewDom) {
    return;
  }
  state.previewOpen = false;
  state.previewFilename = "";
  state.previewCanvas = null;
  previewDom.root.hidden = true;
  previewDom.image.removeAttribute("src");
  document.body.classList.remove("modal-open");
}

function downloadPreviewFallback() {
  if (!state.previewCanvas) {
    return;
  }
  const link = document.createElement("a");
  link.href = state.previewCanvas.toDataURL("image/png");
  link.download = state.previewFilename || "track_viewer.png";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function downloadPreview() {
  if (!state.previewCanvas) {
    return;
  }
  if (typeof state.previewCanvas.toBlob !== "function") {
    downloadPreviewFallback();
    return;
  }
  state.previewCanvas.toBlob((blob) => {
    if (!blob) {
      downloadPreviewFallback();
      return;
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = state.previewFilename || "track_viewer.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }, "image/png");
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
    canvasShell: qs(".canvas-shell", panelEl),
    canvas: qs(".viewerCanvas", panelEl),
    emptyState: qs(".emptyState", panelEl),
    idFilterRoot: qs(".id-filter", panelEl),
    idFilterSummary: qs(".idFilterSummary", panelEl),
    idFilterChips: qs(".idFilterChips", panelEl),
    idFilterEmpty: qs(".idFilterEmpty", panelEl),
    idShowAllBtn: qs(".idShowAllBtn", panelEl),
    idHideAllBtn: qs(".idHideAllBtn", panelEl),
    filterModeButtons: Array.from(panelEl.querySelectorAll(".panelFilterModes [data-filter-mode]")),
    ctx: qs(".viewerCanvas", panelEl).getContext("2d"),
    minFrame: 1,
    maxFrame: 1,
    frame: 1,
    cacheFrames: null,
    lastBoxes: [],
    lastImage: null,
    hasGt: false,
    filterMode: "track",
    trackFilterItems: [],
    gtFilterItems: [],
    hiddenTrackIds: new Set(),
    hiddenGtIds: new Set(),
    isLoading: false,
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
    resetPanelIdFilter(panel);
    loadSequences(panel);
    state.diffFrames = [];
  });

  panel.sequenceSelect.addEventListener("change", () => {
    resetPanelIdFilter(panel);
    if (state.followSequence) {
      followSequence(panel);
    }
    loadFrameRange(panel, false);
    state.diffFrames = [];
  });

  panel.root.addEventListener("click", () => setActivePanel(panel));

  panel.canvasShell.addEventListener("click", () => {
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }
    openPreviewModal(panel);
  });

  panel.idShowAllBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    showAllIds(panel);
  });

  panel.idHideAllBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    hideAllIds(panel);
  });

  panel.filterModeButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      panel.filterMode = button.dataset.filterMode || "track";
      syncPanelFilterModeButtons(panel);
      refreshPanelsDisplay();
    });
  });

  syncPanelFilterModeButtons(panel);
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
    panel.hasGt = Boolean(data.has_gt);
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
  panel.hasGt = Boolean(data.has_gt);
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
  resetPanelIdFilter(targetPanel);
  loadFrameRange(targetPanel, true);
}

function resetPanelIdFilter(panel) {
  panel.trackFilterItems = [];
  panel.gtFilterItems = [];
  panel.hiddenTrackIds.clear();
  panel.hiddenGtIds.clear();
  renderIdFilter(panel);
}

function getEffectiveFilterMode(panel) {
  return panel.filterMode === "gt" && panel.hasGt ? "gt" : "track";
}

function getActiveHiddenSet(panel) {
  return getEffectiveFilterMode(panel) === "gt" ? panel.hiddenGtIds : panel.hiddenTrackIds;
}

function getActiveFilterItems(panel) {
  return getEffectiveFilterMode(panel) === "gt" ? panel.gtFilterItems : panel.trackFilterItems;
}

function getFilterKeyForBox(mode, box) {
  if (mode === "gt") {
    return box.gt_id === undefined || box.gt_id === null
      ? GT_UNMATCHED_KEY
      : String(box.gt_id);
  }
  return String(box.id);
}

function buildTrackFilterItems(entries) {
  const ids = entries
    .map((entry) => entry.box.id)
    .filter((id) => id !== undefined && id !== null);
  return Array.from(new Set(ids))
    .sort((left, right) => Number(left) - Number(right))
    .map((id) => ({ key: String(id), label: `ID ${id}` }));
}

function buildGtFilterItems(entries) {
  const gtIds = [];
  let hasUnmatched = false;
  entries.forEach((entry) => {
    const gtId = entry.box.gt_id;
    if (gtId === undefined || gtId === null) {
      hasUnmatched = true;
      return;
    }
    gtIds.push(gtId);
  });
  const items = Array.from(new Set(gtIds))
    .sort((left, right) => Number(left) - Number(right))
    .map((id) => ({ key: String(id), label: `GT ${id}` }));
  if (hasUnmatched) {
    items.push({ key: GT_UNMATCHED_KEY, label: "未匹配" });
  }
  return items;
}

function getFilteredEntries(panel, boxes, options = {}) {
  const { applyDiff = state.diffOnly, applyManual = true } = options;
  const mode = getEffectiveFilterMode(panel);
  let filtered = boxes
    .filter((box) => Number(box.score ?? 1) >= state.scoreThreshold)
    // Keep thresholdIndex aligned with getCompareInfo(), which applies the same
    // score-only ordering before diff-mode looks up unmatched entries.
    .map((box, thresholdIndex) => ({ box, thresholdIndex }));

  if (applyDiff) {
    const compareInfo = getCompareInfo();
    const unmatched = getUnmatchedSet(panel, compareInfo);
    if (unmatched) {
      filtered = filtered.filter((entry) => unmatched.has(entry.thresholdIndex));
    }
  }

  if (applyManual) {
    const hiddenSet = mode === "gt" ? panel.hiddenGtIds : panel.hiddenTrackIds;
    filtered = filtered.filter(
      (entry) => !hiddenSet.has(getFilterKeyForBox(mode, entry.box))
    );
  }

  return filtered;
}

function renderIdFilter(panel) {
  const requestedGtMode = panel.filterMode === "gt";
  const effectiveMode = getEffectiveFilterMode(panel);
  const items = getActiveFilterItems(panel);
  const hiddenSet = getActiveHiddenSet(panel);
  const hiddenCount = items.filter((item) => hiddenSet.has(item.key)).length;
  if (requestedGtMode && !panel.hasGt) {
    panel.idFilterSummary.textContent = "当前序列无 GT，按 Track ID 过滤";
  } else if (effectiveMode === "gt") {
    panel.idFilterSummary.textContent = items.length
      ? `当前帧 GT: ${items.length} | 已隐藏: ${hiddenCount}`
      : "当前帧 GT: 0";
  } else {
    panel.idFilterSummary.textContent = items.length
      ? `当前帧 ID: ${items.length} | 已隐藏: ${hiddenCount}`
      : "当前帧 ID: 0";
  }
  panel.idFilterChips.innerHTML = "";
  panel.idFilterEmpty.textContent =
    effectiveMode === "gt" ? "当前帧无可选 GT ID" : "当前帧无可选 ID";
  panel.idFilterEmpty.style.display = items.length ? "none" : "block";

  items.forEach((item) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `id-chip${hiddenSet.has(item.key) ? " is-hidden" : ""}`;
    chip.textContent = item.label;
    chip.addEventListener("click", (event) => {
      event.stopPropagation();
      togglePanelId(panel, item.key);
    });
    panel.idFilterChips.appendChild(chip);
  });
}

function syncPanelFilterModeButtons(panel) {
  panel.filterModeButtons.forEach((button) => {
    const mode = button.dataset.filterMode || "track";
    button.classList.toggle("active", mode === panel.filterMode);
  });
}

function updateAvailableIds(panel, boxes) {
  const entries = getFilteredEntries(panel, boxes, { applyManual: false });
  panel.trackFilterItems = buildTrackFilterItems(entries);
  panel.gtFilterItems = panel.hasGt ? buildGtFilterItems(entries) : [];
  renderIdFilter(panel);
}

function paintPanel(panel) {
  if (!panel.lastImage) {
    panel.ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);
    panel.emptyState.style.display = "block";
    panel.countLabel.textContent = `框 ${countVisibleBoxes(panel, panel.lastBoxes)}`;
    return;
  }
  panel.canvas.width = panel.lastImage.naturalWidth;
  panel.canvas.height = panel.lastImage.naturalHeight;
  panel.ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);
  panel.ctx.drawImage(panel.lastImage, 0, 0);
  drawBoxes(panel, panel.lastBoxes);
  panel.emptyState.style.display = "none";
  panel.statusLabel.textContent = `帧 ${panel.frame}`;
  panel.countLabel.textContent = `框 ${countVisibleBoxes(panel, panel.lastBoxes)}`;
}

function refreshPanel(panel) {
  updateAvailableIds(panel, panel.lastBoxes);
  paintPanel(panel);
}

function refreshPanelsDisplay() {
  const panels = window.viewerPanels || [];
  panels.forEach((panel) => refreshPanel(panel));
  updateCompareStats();
}

function togglePanelId(panel, id) {
  const key = String(id);
  const hiddenSet = getActiveHiddenSet(panel);
  if (hiddenSet.has(key)) {
    hiddenSet.delete(key);
  } else {
    hiddenSet.add(key);
  }
  refreshPanel(panel);
}

function showAllIds(panel) {
  const hiddenSet = getActiveHiddenSet(panel);
  if (!hiddenSet.size) {
    return;
  }
  hiddenSet.clear();
  refreshPanel(panel);
}

function hideAllIds(panel) {
  const items = getActiveFilterItems(panel);
  const hiddenSet = getActiveHiddenSet(panel);
  if (!items.length) {
    return;
  }
  items.forEach((item) => hiddenSet.add(item.key));
  refreshPanel(panel);
}

async function renderFrame(panel) {
  const folder = panel.folderSelect.value;
  const sequence = panel.sequenceSelect.value;
  const frame = panel.frame;
  if (!folder || !sequence) {
    return;
  }

  panel.isLoading = true;
  panel.lastImage = null;
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
    panel.hasGt = Boolean(data.has_gt);
    panel.rangeLabel.textContent = `帧范围: ${panel.minFrame} - ${panel.maxFrame}`;
    boxes = data.boxes || [];
  }
  panel.lastBoxes = boxes;
  updateAvailableIds(panel, boxes);

  const imgUrl = `/api/image?folder=${encodeURIComponent(folder)}&sequence=${encodeURIComponent(
    sequence
  )}&frame=${frame}`;

  const image = new Image();
  image.onload = () => {
    panel.isLoading = false;
    panel.lastImage = image;
    refreshPanelsDisplay();
  };
  image.onerror = () => {
    panel.isLoading = false;
    panel.lastImage = null;
    panel.ctx.clearRect(0, 0, panel.canvas.width, panel.canvas.height);
    panel.emptyState.style.display = "block";
    panel.statusLabel.textContent = "找不到图片";
    panel.countLabel.textContent = `框 ${countVisibleBoxes(panel, boxes)}`;
  };
  image.src = imgUrl;
}

function drawBoxes(panel, boxes) {
  if (!boxes.length) {
    panel.countLabel.textContent = "框 0";
    return;
  }
  const filteredEntries = getFilteredEntries(panel, boxes);
  if (!filteredEntries.length) {
    panel.countLabel.textContent = "框 0";
    return;
  }
  const compareInfo = getCompareInfo();
  const unmatched = getUnmatchedSet(panel, compareInfo);
  panel.ctx.lineWidth = Math.max(2, panel.canvas.width / 640);
  panel.ctx.font = `${Math.max(12, panel.canvas.width / 80)}px Space Grotesk`;
  panel.ctx.textBaseline = "top";

  filteredEntries.forEach((entry) => {
    const box = entry.box;
    const score = Number(box.score ?? 1);
    const color = colorForId(box.id || 0);
    const isUnmatched = unmatched ? unmatched.has(entry.thresholdIndex) : false;
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
  if (state.showGtIds && box.gt_id !== undefined && box.gt_id !== null) {
    parts.push(`GT ${box.gt_id}`);
  }
  if (state.showScore) {
    parts.push(score.toFixed(2));
  }
  return parts.join(" ");
}

function countVisibleBoxes(panel, boxes) {
  return getFilteredEntries(panel, boxes).length;
}

function filterBoxes(panel, boxes, options = {}) {
  return getFilteredEntries(panel, boxes, options).map((entry) => entry.box);
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
  panel.hasGt = Boolean(data.has_gt);
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
  previewDom = {
    root: qs("#imageModal"),
    dialog: qs("#imageModalDialog"),
    title: qs("#imageModalTitle"),
    image: qs("#imageModalPreview"),
    saveBtn: qs("#imageSaveBtn"),
    closeBtn: qs("#imageCloseBtn"),
  };

  previewDom.root.addEventListener("click", () => closePreviewModal());
  previewDom.dialog.addEventListener("click", (event) => event.stopPropagation());
  previewDom.saveBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    downloadPreview();
  });
  previewDom.closeBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    closePreviewModal();
  });

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
    refreshPanelsDisplay();
  });

  qs("#gtIdToggle").addEventListener("change", (event) => {
    state.showGtIds = event.target.checked;
    refreshPanelsDisplay();
  });

  qs("#scoreToggle").addEventListener("change", (event) => {
    state.showScore = event.target.checked;
    refreshPanelsDisplay();
  });

  qs("#cacheToggle").addEventListener("change", (event) => {
    state.cacheEnabled = event.target.checked;
    panels.forEach((panel) => loadFrameRange(panel, true));
    updateCompareStats();
  });

  qs("#scoreThreshold").addEventListener("change", (event) => {
    state.scoreThreshold = Number(event.target.value) || 0;
    document.querySelectorAll(".threshold-buttons [data-threshold]").forEach((btn) => {
      const value = Number(btn.dataset.threshold);
      btn.classList.toggle("active", value === state.scoreThreshold);
    });
    refreshPanelsDisplay();
    state.diffFrames = [];
  });

  qs("#iouThreshold").addEventListener("change", (event) => {
    state.diffIou = Number(event.target.value) || 0.5;
    refreshPanelsDisplay();
    state.diffFrames = [];
  });

  qs("#followToggle").addEventListener("change", (event) => {
    state.followSequence = event.target.checked;
  });

  qs("#diffToggle").addEventListener("change", (event) => {
    state.diffOnly = event.target.checked;
    refreshPanelsDisplay();
    state.diffFrames = [];
  });

  document.querySelectorAll(".threshold-buttons [data-threshold]").forEach((button) => {
    button.addEventListener("click", () => {
      const value = Number(button.dataset.threshold);
      qs("#scoreThreshold").value = value.toFixed(2);
      state.scoreThreshold = value;
      document.querySelectorAll(".threshold-buttons [data-threshold]").forEach((btn) => {
        btn.classList.toggle("active", btn === button);
      });
      refreshPanelsDisplay();
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
    if (state.previewOpen) {
      if (event.key === "Escape") {
        event.preventDefault();
        closePreviewModal();
      }
      return;
    }
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) {
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.target.blur();
      } else {
        return;
      }
    }
    const activePanel = window.activePanel || panels[0];
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      stepFrame(activePanel, -1);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      stepFrame(activePanel, 1);
    }
  });

  loadFolders(panels);
  setActivePanel(panels[0]);
}

window.addEventListener("DOMContentLoaded", init);
