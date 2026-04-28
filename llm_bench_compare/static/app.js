/**
 * app.js — Client-side logic for LLM Benchmark Compare
 *
 * Responsibilities:
 *   - Render the benchmark table from initial JSON data
 *   - Apply dynamic multi-dimensional filters (category, size, license, family, open-weights)
 *   - Perform client-side text search
 *   - Sort table columns (ascending / descending)
 *   - Manage up-to-4-model radar chart comparison via Chart.js
 *   - Fetch and display per-model pricing detail in a modal
 *   - Show / hide the cost overlay columns
 */

'use strict';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BENCHMARK_KEYS = JSON.parse(
  document.getElementById('benchmarkKeysData').textContent
);

/** All models loaded from the server-rendered JSON blob. */
const ALL_MODELS = JSON.parse(
  document.getElementById('initialData').textContent
);

/** Colours for radar chart datasets (up to 4). */
const CHART_COLOURS = [
  { border: 'rgba(99, 102, 241, 1)',  background: 'rgba(99, 102, 241, 0.15)'  },
  { border: 'rgba(16, 185, 129, 1)',  background: 'rgba(16, 185, 129, 0.15)'  },
  { border: 'rgba(245, 158, 11, 1)',  background: 'rgba(245, 158, 11, 0.15)'  },
  { border: 'rgba(239, 68, 68, 1)',   background: 'rgba(239, 68, 68, 0.15)'   },
];

const MAX_COMPARE = 4;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** Currently visible / filtered list of model objects. */
let filteredModels = [...ALL_MODELS];

/** Set of model_ids currently selected for radar comparison. */
let selectedForCompare = new Set();

/** Current sort state. */
let sortState = { col: null, asc: false };

/** Reference to the Chart.js instance (singleton). */
let radarChart = null;

/** Whether the cost overlay columns are visible. */
let costOverlayVisible = true;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const tableBody        = document.getElementById('benchTableBody');
const emptyState       = document.getElementById('emptyState');
const visibleCount     = document.getElementById('visibleCount');
const tableFooterText  = document.getElementById('tableFooterText');
const loadingIndicator = document.getElementById('loadingIndicator');
const searchInput      = document.getElementById('searchInput');
const clearFiltersBtn  = document.getElementById('clearFiltersBtn');
const emptyStateClearBtn = document.getElementById('emptyStateClearBtn');
const showChartBtn     = document.getElementById('showChartBtn');
const compareSelection = document.getElementById('compareSelection');
const showCostOverlay  = document.getElementById('showCostOverlay');
const selectAllCompare = document.getElementById('selectAllCompare');

// Modals
const chartModal        = document.getElementById('chartModal');
const closeChartModal   = document.getElementById('closeChartModal');
const closeChartModalFooter = document.getElementById('closeChartModalFooter');
const pricingModal      = document.getElementById('pricingModal');
const pricingModalBody  = document.getElementById('pricingModalBody');
const pricingModalTitle = document.getElementById('pricingModalTitle');
const closePricingModal = document.getElementById('closePricingModal');
const closePricingModalFooter = document.getElementById('closePricingModalFooter');

// Cost columns
const costColHeader = document.getElementById('costColHeader');
const gpuColHeader  = document.getElementById('gpuColHeader');

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/**
 * Format a number to at most 2 decimal places, returning '—' for null/NaN.
 * @param {number|null} val
 * @param {number} [decimals=1]
 * @returns {string}
 */
function fmt(val, decimals = 1) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  return Number(val).toFixed(decimals);
}

/**
 * Format a cost value (USD) with up to 3 significant decimal places.
 * @param {number|null} val
 * @returns {string}
 */
function fmtCost(val) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  if (val < 0.01) return '$' + Number(val).toFixed(4);
  if (val < 1)    return '$' + Number(val).toFixed(3);
  return '$' + Number(val).toFixed(2);
}

/**
 * Return a CSS class name based on a benchmark score (0-100).
 * @param {number|null} val
 * @returns {string}
 */
function scoreClass(val) {
  if (val === null || val === undefined || isNaN(val)) return 'score-na';
  if (val >= 85) return 'score-high';
  if (val >= 65) return 'score-mid';
  return 'score-low';
}

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function escHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Capitalise the first letter of a string.
 * @param {string} s
 * @returns {string}
 */
function capitalise(s) {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Build a license badge HTML snippet.
 * @param {string|null} license
 * @returns {string}
 */
function licenseBadge(license) {
  if (!license) return '<span class="badge badge-na">—</span>';
  let cls = 'badge-commercial';
  if (license === 'Apache-2.0') cls = 'badge-apache';
  else if (license === 'MIT')   cls = 'badge-mit';
  return `<span class="badge ${cls}">${escHtml(license)}</span>`;
}

/**
 * Collect the current active filter values from the DOM.
 * @returns {{ categories: string[], sizes: string[], licenses: string[], families: string[], openWeights: boolean, search: string }}
 */
function getActiveFilters() {
  const checked = (name) =>
    [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((el) => el.value);

  return {
    categories:  checked('category'),
    sizes:       checked('size'),
    licenses:    checked('license'),
    families:    checked('family'),
    openWeights: document.getElementById('openWeightsFilter').checked,
    search:      searchInput.value.trim().toLowerCase(),
  };
}

/**
 * Apply the active filters to ALL_MODELS and return a filtered array.
 * @returns {object[]}
 */
function applyClientFilters() {
  const f = getActiveFilters();

  return ALL_MODELS.filter((model) => {
    // Search
    if (f.search) {
      const haystack = [
        model.display_name,
        model.family,
        model.license,
        model.parameter_size_bucket,
        ...(model.task_categories || []),
      ].join(' ').toLowerCase();
      if (!haystack.includes(f.search)) return false;
    }

    // Task category (OR logic)
    if (f.categories.length > 0) {
      const modelCats = model.task_categories || [];
      if (!f.categories.some((c) => modelCats.includes(c))) return false;
    }

    // Parameter size bucket (OR logic)
    if (f.sizes.length > 0) {
      if (!f.sizes.includes(model.parameter_size_bucket)) return false;
    }

    // License (OR logic)
    if (f.licenses.length > 0) {
      if (!f.licenses.includes(model.license)) return false;
    }

    // Family (OR logic)
    if (f.families.length > 0) {
      if (!f.families.includes(model.family)) return false;
    }

    // Open weights
    if (f.openWeights && !model.open_weights) return false;

    return true;
  });
}

/**
 * Sort an array of model objects by the current sortState.
 * @param {object[]} models
 * @returns {object[]}
 */
function sortModels(models) {
  if (!sortState.col) return models;

  return [...models].sort((a, b) => {
    let va = a[sortState.col];
    let vb = b[sortState.col];

    // Nulls / NaN always last
    const aNull = va === null || va === undefined || va !== va;
    const bNull = vb === null || vb === undefined || vb !== vb;
    if (aNull && bNull) return 0;
    if (aNull) return 1;
    if (bNull) return -1;

    if (typeof va === 'string') {
      va = va.toLowerCase();
      vb = (vb || '').toLowerCase();
    }

    if (va < vb) return sortState.asc ? -1 : 1;
    if (va > vb) return sortState.asc ? 1 : -1;
    return 0;
  });
}

// ---------------------------------------------------------------------------
// Table rendering
// ---------------------------------------------------------------------------

/**
 * Build and inject all table rows for the given model array.
 * @param {object[]} models
 */
function renderTableRows(models) {
  if (models.length === 0) {
    tableBody.innerHTML = '';
    emptyState.classList.remove('hidden');
    return;
  }
  emptyState.classList.add('hidden');

  const fragments = models.map((model) => buildRow(model));
  tableBody.innerHTML = fragments.join('');

  // Re-attach compare checkbox listeners
  tableBody.querySelectorAll('.compare-checkbox').forEach((cb) => {
    cb.addEventListener('change', onCompareCheckboxChange);
    // Reflect current selection state
    cb.checked = selectedForCompare.has(cb.dataset.modelId);
    // Disable unchecked boxes if we're at the limit
    if (selectedForCompare.size >= MAX_COMPARE && !cb.checked) {
      cb.disabled = true;
    }
  });

  // Pricing detail buttons
  tableBody.querySelectorAll('.pricing-detail-btn').forEach((btn) => {
    btn.addEventListener('click', onPricingDetailClick);
  });
}

/**
 * Build the HTML string for a single table row.
 * @param {object} model
 * @returns {string}
 */
function buildRow(model) {
  const isSelected = selectedForCompare.has(model.model_id);
  const atLimit = selectedForCompare.size >= MAX_COMPARE && !isSelected;

  // Size label
  const sizeLabel = model.parameter_size_b !== null && model.parameter_size_b !== undefined
    ? `${model.parameter_size_b}B`
    : model.parameter_size_bucket || '—';

  // Benchmark score cells
  const benchCells = BENCHMARK_KEYS.map((key) => {
    const val = model[`benchmark_${key}`];
    const cls = scoreClass(val);
    return `<td class="col-bench score-cell ${cls}">${fmt(val)}</td>`;
  }).join('');

  // Cost cells
  const apiCostHtml = buildApiCostCell(model);
  const gpuCostHtml = buildGpuCostCell(model);

  const costColVisible = costOverlayVisible ? '' : ' hidden';

  return `
    <tr class="bench-row${isSelected ? ' row-selected' : ''}" data-model-id="${escHtml(model.model_id)}">
      <td class="col-compare">
        <input
          type="checkbox"
          class="compare-checkbox"
          data-model-id="${escHtml(model.model_id)}"
          data-display-name="${escHtml(model.display_name)}"
          ${isSelected ? 'checked' : ''}
          ${atLimit ? 'disabled' : ''}
          title="Add to radar comparison"
        />
      </td>
      <td class="col-model">
        <span class="model-name">${escHtml(model.display_name)}</span>
        ${model.open_weights
          ? '<span class="badge badge-open" title="Open weights">&#128275;</span>'
          : '<span class="badge badge-closed" title="Closed weights">&#128274;</span>'}
      </td>
      <td class="col-family">
        <span class="family-tag family-${escHtml((model.family || '').toLowerCase().replace(/[^a-z0-9]/g, '-'))}">
          ${escHtml(model.family || '—')}
        </span>
      </td>
      <td class="col-size">${escHtml(sizeLabel)}</td>
      ${benchCells}
      <td class="col-cost cost-col${costColVisible}">${apiCostHtml}</td>
      <td class="col-gpu cost-col${costColVisible}">${gpuCostHtml}</td>
      <td class="col-license">${licenseBadge(model.license)}</td>
    </tr>`;
}

/**
 * Build the API cost cell HTML (cheapest input+output).
 * @param {object} model
 * @returns {string}
 */
function buildApiCostCell(model) {
  const inp = model.cheapest_input_per_1m;
  const out = model.cheapest_output_per_1m;
  if ((inp === null || inp === undefined) && (out === null || out === undefined)) {
    return '<span class="cost-na">—</span>';
  }
  return `
    <div class="cost-badge">
      <button class="pricing-detail-btn" data-model-id="${escHtml(model.model_id)}" data-display-name="${escHtml(model.display_name)}" title="View full pricing detail">
        <span class="cost-in" title="Input per 1M tokens">${fmtCost(inp)}</span>
        <span class="cost-sep">/</span>
        <span class="cost-out" title="Output per 1M tokens">${fmtCost(out)}</span>
      </button>
    </div>`;
}

/**
 * Build the self-hosted GPU cost cell HTML.
 * @param {object} model
 * @returns {string}
 */
function buildGpuCostCell(model) {
  const cost = model.self_hosted_hourly_usd;
  if (cost === null || cost === undefined) {
    return '<span class="cost-na">—</span>';
  }
  const setup = model.self_hosted_gpu_setup || '';
  return `<span class="gpu-cost" title="${escHtml(setup)}">${fmtCost(cost)}/hr</span>`;
}

// ---------------------------------------------------------------------------
// Filter & render pipeline
// ---------------------------------------------------------------------------

/**
 * Re-apply all active filters, sort, and re-render the table.
 */
function refreshTable() {
  filteredModels = sortModels(applyClientFilters());

  renderTableRows(filteredModels);
  updateCountDisplay();
  updateSortIcons();
}

/**
 * Update the "Showing X of Y" count display.
 */
function updateCountDisplay() {
  visibleCount.textContent = filteredModels.length;
  const total = ALL_MODELS.length;
  tableFooterText.textContent =
    filteredModels.length === total
      ? `All ${total} models`
      : `${filteredModels.length} of ${total} models match current filters`;
}

/**
 * Update sort icon indicators in the table header.
 */
function updateSortIcons() {
  document.querySelectorAll('.sortable').forEach((th) => {
    const icon = th.querySelector('.sort-icon');
    if (!icon) return;
    if (th.dataset.col === sortState.col) {
      icon.textContent = sortState.asc ? ' ▲' : ' ▼';
      th.classList.add('sorted');
    } else {
      icon.textContent = '';
      th.classList.remove('sorted');
    }
  });
}

/**
 * Clear all active filters and reset the UI.
 */
function clearAllFilters() {
  document.querySelectorAll(
    'input[name="category"], input[name="size"], input[name="license"], input[name="family"], #openWeightsFilter'
  ).forEach((el) => { el.checked = false; });
  searchInput.value = '';
  sortState = { col: null, asc: false };
  refreshTable();
}

// ---------------------------------------------------------------------------
// Sort handling
// ---------------------------------------------------------------------------

/**
 * Handle a click on a sortable column header.
 * @param {Event} e
 */
function onSortHeaderClick(e) {
  const th = e.currentTarget;
  const col = th.dataset.col;
  if (!col) return;

  if (sortState.col === col) {
    sortState.asc = !sortState.asc;
  } else {
    sortState.col = col;
    // Numbers default to descending (highest first); strings default to ascending
    sortState.asc = th.dataset.type === 'string';
  }
  refreshTable();
}

// ---------------------------------------------------------------------------
// Radar chart comparison
// ---------------------------------------------------------------------------

/**
 * Handle a compare checkbox change.
 * @param {Event} e
 */
function onCompareCheckboxChange(e) {
  const cb = e.currentTarget;
  const modelId = cb.dataset.modelId;
  const displayName = cb.dataset.displayName;

  if (cb.checked) {
    if (selectedForCompare.size >= MAX_COMPARE) {
      cb.checked = false;
      return;
    }
    selectedForCompare.add(modelId);
  } else {
    selectedForCompare.delete(modelId);
  }

  updateCompareUI();

  // Highlight selected rows
  tableBody.querySelectorAll('.bench-row').forEach((row) => {
    const mid = row.dataset.modelId;
    row.classList.toggle('row-selected', selectedForCompare.has(mid));
  });

  // Disable unchecked checkboxes at limit
  tableBody.querySelectorAll('.compare-checkbox').forEach((box) => {
    if (!box.checked) {
      box.disabled = selectedForCompare.size >= MAX_COMPARE;
    }
  });
}

/**
 * Update the compare selection chips and Show Chart button.
 */
function updateCompareUI() {
  showChartBtn.disabled = selectedForCompare.size < 1;

  // Build chip list
  compareSelection.innerHTML = '';
  selectedForCompare.forEach((modelId) => {
    const model = ALL_MODELS.find((m) => m.model_id === modelId);
    if (!model) return;
    const chip = document.createElement('div');
    chip.className = 'compare-chip';
    chip.innerHTML = `
      <span class="chip-label">${escHtml(model.display_name)}</span>
      <button class="chip-remove" data-model-id="${escHtml(modelId)}" title="Remove">&#10005;</button>`;
    chip.querySelector('.chip-remove').addEventListener('click', () => {
      selectedForCompare.delete(modelId);
      // Uncheck the corresponding row checkbox
      const rowCb = tableBody.querySelector(`.compare-checkbox[data-model-id="${modelId}"]`);
      if (rowCb) rowCb.checked = false;
      // Re-enable checkboxes
      tableBody.querySelectorAll('.compare-checkbox').forEach((box) => {
        if (!box.checked) box.disabled = false;
      });
      // Remove selected class from row
      const row = tableBody.querySelector(`.bench-row[data-model-id="${modelId}"]`);
      if (row) row.classList.remove('row-selected');

      updateCompareUI();
    });
    compareSelection.appendChild(chip);
  });
}

/**
 * Handle "Select All" checkbox for radar comparison.
 */
function onSelectAllChange() {
  const checked = selectAllCompare.checked;
  if (checked) {
    // Select up to MAX_COMPARE from filtered models
    const toSelect = filteredModels.slice(0, MAX_COMPARE);
    selectedForCompare.clear();
    toSelect.forEach((m) => selectedForCompare.add(m.model_id));
  } else {
    selectedForCompare.clear();
  }
  updateCompareUI();
  // Re-render to reflect updated checkboxes
  renderTableRows(filteredModels);
}

/**
 * Open the radar chart modal and render the chart.
 */
function showRadarChart() {
  const modelIds = [...selectedForCompare];
  if (modelIds.length === 0) return;

  const models = modelIds
    .map((id) => ALL_MODELS.find((m) => m.model_id === id))
    .filter(Boolean);

  const labels = BENCHMARK_KEYS.map((k) => k.toUpperCase());

  const datasets = models.map((model, i) => {
    const colour = CHART_COLOURS[i % CHART_COLOURS.length];
    const data = BENCHMARK_KEYS.map((key) => {
      const val = model[`benchmark_${key}`];
      return val !== null && val !== undefined ? val : 0;
    });
    return {
      label: model.display_name,
      data,
      borderColor: colour.border,
      backgroundColor: colour.background,
      borderWidth: 2,
      pointBackgroundColor: colour.border,
      pointRadius: 4,
    };
  });

  // Destroy previous chart instance
  if (radarChart) {
    radarChart.destroy();
    radarChart = null;
  }

  const ctx = document.getElementById('radarChart').getContext('2d');
  radarChart = new Chart(ctx, {
    type: 'radar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: {
            stepSize: 20,
            font: { size: 11 },
            backdropColor: 'transparent',
          },
          pointLabels: {
            font: { size: 13, weight: 'bold' },
          },
          grid: { color: 'rgba(0,0,0,0.08)' },
          angleLines: { color: 'rgba(0,0,0,0.08)' },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(1)}`,
          },
        },
      },
    },
  });

  // Build legend
  const legend = document.getElementById('chartLegend');
  legend.innerHTML = models.map((model, i) => {
    const colour = CHART_COLOURS[i % CHART_COLOURS.length];
    return `
      <div class="legend-item">
        <span class="legend-swatch" style="background:${colour.border}"></span>
        <span class="legend-label">${escHtml(model.display_name)}</span>
      </div>`;
  }).join('');

  chartModal.classList.remove('hidden');
  document.body.classList.add('modal-open');
}

/**
 * Close the radar chart modal.
 */
function closeRadarChart() {
  chartModal.classList.add('hidden');
  document.body.classList.remove('modal-open');
}

// ---------------------------------------------------------------------------
// Pricing detail modal
// ---------------------------------------------------------------------------

/**
 * Handle a click on a pricing detail button in the table.
 * @param {Event} e
 */
function onPricingDetailClick(e) {
  const btn = e.currentTarget;
  const modelId = btn.dataset.modelId;
  const displayName = btn.dataset.displayName;
  openPricingModal(modelId, displayName);
}

/**
 * Open the pricing modal and fetch detail from the API.
 * @param {string} modelId
 * @param {string} displayName
 */
function openPricingModal(modelId, displayName) {
  pricingModalTitle.textContent = `Pricing — ${displayName}`;
  pricingModalBody.innerHTML = '<div class="pricing-loading">Loading pricing data…</div>';
  pricingModal.classList.remove('hidden');
  document.body.classList.add('modal-open');

  fetch(`/api/pricing/${encodeURIComponent(modelId)}`)
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((data) => renderPricingDetail(data))
    .catch((err) => {
      pricingModalBody.innerHTML =
        `<p class="pricing-error">Failed to load pricing data: ${escHtml(err.message)}</p>`;
    });
}

/**
 * Render the pricing detail response into the modal.
 * @param {object} data
 */
function renderPricingDetail(data) {
  const providers = data.api_providers || [];
  const sh = data.self_hosted || {};

  let html = '';

  // API Providers table
  if (providers.length > 0) {
    html += '<h3 class="pricing-section-title">API Providers</h3>';
    html += '<div class="pricing-table-wrapper"><table class="pricing-table">';
    html += '<thead><tr><th>Provider</th><th>Input / 1M tok</th><th>Output / 1M tok</th><th>Notes</th></tr></thead>';
    html += '<tbody>';
    providers.forEach((p) => {
      const inp = p.input_per_1m !== null ? fmtCost(p.input_per_1m) : '—';
      const out = p.output_per_1m !== null ? fmtCost(p.output_per_1m) : '—';
      const notes = p.notes || '—';
      const providerName = p.provider_url
        ? `<a href="${escHtml(p.provider_url)}" target="_blank" rel="noopener noreferrer">${escHtml(p.provider)}</a>`
        : escHtml(p.provider || '—');
      html += `<tr><td>${providerName}</td><td class="price-cell">${escHtml(inp)}</td><td class="price-cell">${escHtml(out)}</td><td class="notes-cell">${escHtml(notes)}</td></tr>`;
    });
    html += '</tbody></table></div>';
  } else {
    html += '<p class="pricing-none">No public API pricing data available for this model.</p>';
  }

  // Self-hosted
  html += '<h3 class="pricing-section-title">Self-Hosted Estimate</h3>';
  if (sh.hourly_cost_usd !== null && sh.hourly_cost_usd !== undefined) {
    html += `
      <div class="sh-grid">
        <div class="sh-card">
          <div class="sh-card-label">Estimated GPU Cost</div>
          <div class="sh-card-value">${fmtCost(sh.hourly_cost_usd)}<span class="sh-unit">/hr</span></div>
        </div>
        <div class="sh-card">
          <div class="sh-card-label">Min VRAM</div>
          <div class="sh-card-value">${sh.min_vram_gb !== null ? sh.min_vram_gb + ' GB' : '—'}</div>
        </div>
        <div class="sh-card">
          <div class="sh-card-label">Throughput (approx)</div>
          <div class="sh-card-value">${sh.throughput_tps !== null ? sh.throughput_tps + ' tok/s' : '—'}</div>
        </div>
      </div>`;
    if (sh.gpu_setup) {
      html += `<p class="sh-setup"><strong>Recommended setup:</strong> ${escHtml(sh.gpu_setup)}</p>`;
    }
  } else {
    html += '<p class="pricing-none">No self-hosted cost estimate available.</p>';
  }

  pricingModalBody.innerHTML = html;
}

/**
 * Close the pricing modal.
 */
function closePricingModalFn() {
  pricingModal.classList.add('hidden');
  document.body.classList.remove('modal-open');
}

// ---------------------------------------------------------------------------
// Cost overlay toggle
// ---------------------------------------------------------------------------

/**
 * Show or hide the API cost and self-hosted GPU columns.
 */
function updateCostOverlayVisibility() {
  costOverlayVisible = showCostOverlay.checked;
  const costCols = document.querySelectorAll('.cost-col');
  costCols.forEach((el) => {
    el.classList.toggle('hidden', !costOverlayVisible);
  });
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

function wireEvents() {
  // Filter checkboxes
  document.querySelectorAll(
    'input[name="category"], input[name="size"], input[name="license"], input[name="family"], #openWeightsFilter'
  ).forEach((el) => el.addEventListener('change', refreshTable));

  // Search
  searchInput.addEventListener('input', refreshTable);

  // Clear buttons
  clearFiltersBtn.addEventListener('click', clearAllFilters);
  emptyStateClearBtn.addEventListener('click', clearAllFilters);

  // Sort headers
  document.querySelectorAll('.sortable').forEach((th) => {
    th.addEventListener('click', onSortHeaderClick);
  });

  // Radar chart
  showChartBtn.addEventListener('click', showRadarChart);
  closeChartModal.addEventListener('click', closeRadarChart);
  closeChartModalFooter.addEventListener('click', closeRadarChart);
  chartModal.addEventListener('click', (e) => {
    if (e.target === chartModal) closeRadarChart();
  });

  // Pricing modal close
  closePricingModal.addEventListener('click', closePricingModalFn);
  closePricingModalFooter.addEventListener('click', closePricingModalFn);
  pricingModal.addEventListener('click', (e) => {
    if (e.target === pricingModal) closePricingModalFn();
  });

  // Cost overlay
  showCostOverlay.addEventListener('change', updateCostOverlayVisibility);

  // Select all
  selectAllCompare.addEventListener('change', onSelectAllChange);

  // Keyboard: close modals on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!chartModal.classList.contains('hidden')) closeRadarChart();
      if (!pricingModal.classList.contains('hidden')) closePricingModalFn();
    }
  });
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

function init() {
  wireEvents();
  refreshTable();
}

document.addEventListener('DOMContentLoaded', init);
