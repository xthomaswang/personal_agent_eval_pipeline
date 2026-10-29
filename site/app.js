'use strict';

const DATA = window.BENCHMARK_DATA;
const REPO = DATA.repoUrl;
const main = document.getElementById('main');
const dialog = document.getElementById('result-dialog');
const state = { view: 'ability', query: '', sort: 'Ranking set', ascending: false, archive: 'aider12-v0', suite: 'capability', component: null, compareModel: null, machine: DATA.machines[0]?.id ?? '', selectedProtocols: {}, uiMode: 'replica:vision', readScoring: 'strict' };
const HARNESS_LABELS = { claude_code: 'Claude Code', hermes: 'Hermes', localmelo: 'localmelo' };
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const countValue = stats => !stats?.runs ? '—' : `${stats.passed}/${stats.runs}${stats.runs < stats.expected ? ' (partial)' : ''}`;
const isCapability = id => /^capability-\d+$/.test(id ?? '');
const metric = (row, name) => {
  if (row._ui && name.startsWith('UI ')) {
    const key = { 'UI click': 'click', 'UI fill': 'fill', 'UI read': state.readScoring === 'strict' ? 'read_strict' : 'read_lenient' }[name];
    const pair = row._ui[state.uiMode]?.[key];
    return pair ? `${pair[0]}/${pair[1]}` : '—';
  }
  return row.metrics.find(item => item.name === name)?.value ?? '—';
};
const fraction = value => { const match = /^(\d+)\/(\d+)/.exec(value); return match ? { pass: Number(match[1]), total: Number(match[2]), pct: 100 * Number(match[1]) / Number(match[2]) } : null; };
const sourceLink = (path, tree = false) => `${REPO}/${tree ? 'tree' : 'blob'}/main/${path}`;
const external = (path, label, tree = false, className = '') => `<a class="${className}" href="${sourceLink(path, tree)}" target="_blank" rel="noreferrer">${label} ↗</a>`;
const shortModel = row => row.model.replace(/ oQ\S+(?: \+ MTP)?$/, '');
const groupsForView = () => DATA.protocols.filter(group => group.machine === state.machine && group.kind === state.view).sort((a, b) => b.latest.localeCompare(a.latest) || a.id.localeCompare(b.id));
const protocol = () => groupsForView().find(group => group.id === state.selectedProtocols[state.view])?.id ?? groupsForView()[0]?.id;
const rawRows = () => DATA.rows.filter(row => row.protocol === protocol() && row.system.machine === state.machine);
const machineLabel = () => DATA.machines.find(machine => machine.id === state.machine)?.label ?? state.machine;
const UI_INPUTS = { 'replica:vision': 'Screenshot', 'replica:vision+ocr': 'Screenshot + OCR', 'replica:ax': 'AX list', 'capture:vision': 'Original capture' };
const VIEW_LABELS = { ability: 'Agent ability', ui: 'UI ability', speed: 'Serving performance', archive: 'Historical results', memory: 'Evolution ability (memory)' };

const partial = row => row.metrics.some(item => item.value.includes('(partial)'));
const flag = row => Boolean(row.notes?.includes('minor disturbance'));
const scoreNumber = (row, name) => {
  const value = metric(row, name);
  if (value.includes('partial') || value === '—') return null;
  const f = fraction(value);
  return f ? f.pct : Number.parseFloat(value);
};

function brandStats() {
  return `<div class="hero-stats"><span><b>${DATA.buildCount}</b> model builds</span><i class="divider"></i><span><b>${DATA.harnessCount}</b> agent harnesses</span><i class="divider"></i><span><b>${DATA.machineCount}</b> Apple-silicon ${DATA.machineCount === 1 ? 'Mac' : 'Macs'}</span><span class="quiet-meta updated">UPDATED ${(DATA.latest ?? 'No results').replaceAll('-', '.')}</span></div>`;
}

function benchmarkPage() {
  const views = Object.entries(VIEW_LABELS).filter(([key]) => DATA.protocols.some(group => group.machine === state.machine && group.kind === key));
  if (!views.some(([key]) => key === state.view) && views.length) state.view = views[0][0];
  // The harness comparison follows the selected agent-ability protocol.
  const comparison = state.view === 'ability' && capabilityRows().length ? `
    <section class="container harness-section" aria-labelledby="harness-heading">
      <div class="comparison-heading"><div><h2 id="harness-heading">Harness comparison</h2><span class="quiet-meta">${escapeHTML(protocol())} · ${capabilityRows().reduce((date, row) => row.date > date ? row.date : date, '')} · same model, same tasks</span></div>
      <label class="model-picker">Model<select id="comparison-model" aria-label="Model for harness comparison">${capabilityRows().map(row => `<option value="${escapeHTML(row._dir)}" ${comparisonModel() === row ? 'selected' : ''}>${escapeHTML(row.model)}</option>`).join('')}</select></label></div>
      <div id="harness-results"></div>
    </section>` : '';
  return `<section class="hero"><div class="container"><div class="eyebrow"><span class="live-dot"></span>Open benchmark · Apple silicon</div><h1>Personal AI Agent<br><span>Benchmark.</span></h1>${brandStats()}</div></section>
    <section class="container results-section" aria-labelledby="results-heading">
      <div class="section-heading"><h2 id="results-heading">Results</h2><span class="snapshot">${DATA.resultCount} PUBLISHED RESULTS</span></div>
      <div class="tabs" role="tablist" aria-label="Benchmark results">${views.map(([key, label]) => `<button class="tab" id="tab-${key}" role="tab" aria-controls="results-panel" data-view="${key}" aria-selected="${state.view === key}">${label}</button>`).join('')}</div>
      <div id="results-panel" role="tabpanel" aria-labelledby="tab-${state.view}">
        <div class="table-toolbar"><div class="scope" id="result-scope"></div><div class="result-filters"><label class="search"><span aria-hidden="true">⌕</span><input id="model-search" type="search" aria-label="Search models" placeholder="Search models…" value="${escapeHTML(state.query)}"></label></div></div>
        <div id="table-results"></div>
      </div>
    </section>${comparison}
    <section class="container measurement-section"><div class="section-kicker">Evaluation suites</div><div class="measurement-grid">
      <div class="measurement"><div class="index">01 /</div><h3><a href="pipeline.html#capability">Agent ability ↗</a></h3><p>Coding, filesystem tasks, and separately scored UI probes.</p></div>
      <div class="measurement"><div class="index">02 /</div><h3><a href="pipeline.html#speed">Serving performance ↗</a></h3><p>Latency, long-context decoding, and server memory.</p></div>
      <div class="measurement"><div class="index">03 /</div><h3><a href="pipeline.html#memory">Evolution ability (memory) ↗</a></h3><span class="subtle-tag">IN DEVELOPMENT</span></div>
    </div></section>`;
}

function scoreCell(value, missingLabel = 'Not evaluated') {
  const f = fraction(value);
  if (value === '—') return `<span class="score" aria-label="No score">—</span><div class="score-detail">${missingLabel}</div>`;
  if (value.includes('(partial)')) return `<span class="score">${f ? `${f.pass}/${f.total}` : escapeHTML(value)}</span><div class="score-detail">Partial result</div>`;
  if (f) return `<span class="score">${f.pct.toFixed(1)}<span class="score-unit">%</span></span><div class="score-detail">${f.pass} / ${f.total} passed</div>`;
  const match = /^([\d.,]+)\s*(.*)$/.exec(value);
  return match ? `<span class="score">${match[1]}<span class="score-unit">${escapeHTML(match[2])}</span></span>` : escapeHTML(value);
}

function tableColumns() {
  const id = protocol();
  if (state.view === 'ui' && rawRows().some(row => row._ui)) return [['UI click', 'Click'], ['UI fill', 'Fill a form'], ['UI read', `Read · ${state.readScoring}`]];
  if (isCapability(id)) return [['Ranking set', 'Task completion'], ['Gate set', 'Gate set'], ['M7 tool-call cases', 'Tool use']];
  if (id === 'speed-1') return [['Cached-prefix TTFT', 'Cached TTFT'], ['Decode 64K', 'Decode @ 64K'], ['Peak server memory', 'Peak memory']];
  if (id === 'aider12-v0') return [['Total', 'Task completion'], ['Claude Code', 'Claude Code'], ['Hermes', 'Hermes'], ['localmelo', 'localmelo']];
  if (id === 'quant-fidelity-v0') return [['KL, all stable positions', 'KL divergence'], ['Top-1 agreement', 'Top-1 agreement'], ['Screenshot codes read', 'Screenshot reading']];
  return (groupsForView().find(group => group.id === id)?.metrics ?? []).map(name => [name, name]);
}

function capabilityRows() {
  return DATA.rows.filter(row => isCapability(protocol()) && row.protocol === protocol() && row.system.machine === state.machine);
}

function comparisonModel() {
  const rows = capabilityRows();
  return rows.find(row => row._dir === state.compareModel) ?? rows.find(row => row.reference) ?? rows[0];
}

function harnessStatus(stats) {
  if (!stats?.gate.runs && !stats?.ranking.runs) return ['unmeasured', 'Not evaluated'];
  if (stats.gate.runs < stats.gate.expected) return ['partial', 'Partial gate'];
  if (!stats.ranking.runs) return ['partial', 'Gate only'];
  if (stats.ranking.runs < stats.ranking.expected) return ['partial', 'Partial ranking'];
  return ['', 'Complete'];
}

function renderHarnessComparison() {
  const target = document.getElementById('harness-results');
  const row = comparisonModel();
  if (!target || !row) return;
  target.innerHTML = `<div class="results-table-wrap">
    <table class="results-table harness-table">
      <caption class="screen-reader">Claude Code, Hermes, and localmelo on ${escapeHTML(row.model)}. Ranking-set completion, gate-set completion, median model time, and median model calls.</caption>
      <colgroup><col class="harness-column"><col><col><col><col></colgroup>
      <thead><tr><th scope="col">Harness</th><th scope="col">Task completion</th><th scope="col">Gate set</th><th scope="col">Model time</th><th scope="col">Model calls</th></tr></thead>
      <tbody>${Object.entries(HARNESS_LABELS).map(([key, label]) => {
        const stats = row._harnesses?.[key];
        const [statusClass, statusLabel] = harnessStatus(stats);
        const time = stats?.ranking.modelTimeMedian;
        const calls = stats?.ranking.modelCallsMedian;
        return `<tr><th scope="row"><strong class="harness-name">${label}</strong><span class="status harness-status ${statusClass}">${statusLabel}</span></th>
          <td class="metric-cell">${scoreCell(countValue(stats?.ranking))}</td>
          <td class="metric-cell">${scoreCell(countValue(stats?.gate))}</td>
          <td class="metric-cell">${scoreCell(time == null ? '—' : `${time.toFixed(1)} s`)}</td>
          <td class="metric-cell">${scoreCell(calls == null ? '—' : String(calls))}</td></tr>`;
      }).join('')}</tbody>
    </table></div>
    <div class="table-note"><p>22 ranking tasks per harness. Time and calls are medians across valid ranking tasks, including failures; gate tasks are separate.</p>${external(row._harness_source, 'Source records')}</div>
    <span class="screen-reader" role="status">Comparing three harnesses on ${escapeHTML(row.model)}</span>`;
}

function resultStatus(row) {
  if (state.view === 'archive') return ['flagged', 'Historical'];
  return partial(row) ? ['partial', 'Partial run'] : flag(row) ? ['flagged', 'Flagged timing'] : ['', 'Complete'];
}

function renderResults() {
  const scope = document.getElementById('result-scope');
  if (!scope) return;
  const rows = rawRows();
  const columns = tableColumns();
  const groups = groupsForView();
  const machineControl = DATA.machines.length > 1
    ? `<label class="result-control">Machine<select id="result-machine" aria-label="Benchmark machine">${DATA.machines.map(machine => `<option value="${escapeHTML(machine.id)}" ${state.machine === machine.id ? 'selected' : ''}>${escapeHTML(machine.label)}</option>`).join('')}</select></label>`
    : `<span class="chip">${escapeHTML(machineLabel())}</span>`;
  const protocolControl = groups.length > 1
    ? `<label class="result-control">Protocol<select id="result-protocol" aria-label="Benchmark protocol">${groups.map(group => `<option value="${escapeHTML(group.id)}" ${protocol() === group.id ? 'selected' : ''}>${escapeHTML(group.id)}</option>`).join('')}</select></label>`
    : `<span class="chip">${escapeHTML(protocol() ?? 'No results')}</span>`;
  const uiControl = state.view === 'ui' && rows.some(row => row._ui) ? `<div class="ui-controls">
    <label class="result-control">Input<select id="ui-input" aria-label="UI input">${Object.entries(UI_INPUTS).map(([key, label]) => `<option value="${key}" ${state.uiMode === key ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
    <label class="result-control">Reads<select id="ui-scoring" aria-label="Read scoring"><option value="strict" ${state.readScoring === 'strict' ? 'selected' : ''}>Strict</option><option value="lenient" ${state.readScoring === 'lenient' ? 'selected' : ''}>Lenient</option></select></label></div>` : '';
  scope.innerHTML = `${machineControl}${protocolControl}${uiControl}`;
  const filtered = [...rows].sort((a, b) => {
    const x = scoreNumber(a, state.sort), y = scoreNumber(b, state.sort);
    if (x === null || Number.isNaN(x)) return y === null || Number.isNaN(y) ? a.model.localeCompare(b.model) : 1;
    if (y === null || Number.isNaN(y)) return -1;
    return (state.ascending ? x - y : y - x) || a.model.localeCompare(b.model);
  }).filter(row => row.model.toLowerCase().includes(state.query.toLowerCase()));
  const uiMeta = rows[0]?._ui_meta;
  const uiCohort = uiMeta ? `<div class="ui-cohort">${uiMeta.pages ? `${uiMeta.pages} pages · ` : ''}${state.uiMode === 'capture:vision' ? uiMeta.items.read : uiMeta.totalItems} ${state.uiMode === 'capture:vision' ? 'reading items' : 'items'} × ${uiMeta.samples} samples · audited gold</div>` : '';
  const note = state.view === 'ui' && uiMeta
    ? `${state.uiMode === 'capture:vision' ? `Original captures: ${uiMeta.items.read} reading items; click and fill are not scored.` : `Click: ${uiMeta.items.click} items; fill: ${uiMeta.items.fill} forms; read: ${uiMeta.items.read} items.`} ${state.readScoring === 'strict' ? 'Strict reads use exact-match scoring.' : 'Lenient reads additionally normalize formatting.'} Counts are item-samples.`
    : isCapability(protocol())
      ? '22 ranking tasks × 3 harnesses. Partial runs unranked; statistical significance assessed separately.'
      : state.view === 'speed'
        ? 'Cached TTFT: ~15K-token prefix. Minor timing disturbances are flagged; full metrics are in model details.'
        : state.view === 'archive'
          ? 'Earlier patched-server records; separate protocols, not reproducible from this repository alone.'
          : 'Results share a machine and a versioned protocol. Open a model for run evidence.';
  document.getElementById('table-results').innerHTML = `${uiCohort}
    <div class="results-table-wrap">
      <table class="results-table" style="min-width:${210 + columns.length * 124 + 100 + 42}px">
        <caption class="screen-reader">${escapeHTML(rows[0]?.title ?? 'Benchmark results')}. ${escapeHTML(protocol() ?? '')} on ${escapeHTML(state.machine)}.</caption>
        <colgroup><col class="model-column">${columns.map(() => '<col class="metric-column">').join('')}<col class="status-column"><col class="details-column"></colgroup>
        <thead><tr><th scope="col">Model / build</th>${columns.map(([key, label]) => `
          <th scope="col" aria-sort="${state.sort === key ? state.ascending ? 'ascending' : 'descending' : 'none'}"><button class="sort-button" data-sort="${escapeHTML(key)}" aria-label="Sort by ${escapeHTML(label)}">${label}<span aria-hidden="true">${state.sort === key ? state.ascending ? '▴' : '▾' : '↕'}</span></button></th>`).join('')}
          <th scope="col">Run status</th><th scope="col"><span class="screen-reader">Details</span></th></tr></thead>
        <tbody>${filtered.map(row => {
          const index = DATA.rows.indexOf(row);
          const [statusClass, statusLabel] = resultStatus(row);
          const modelLabel = `${row.quant.replace('oq', 'oQ')}${row.model.includes('MTP') ? ' + MTP' : ''}`;
          return `<tr><td><button class="model-button" data-result="${index}" aria-label="View ${escapeHTML(row.model)} result details">${escapeHTML(shortModel(row))}</button><div class="model-sub">${escapeHTML(modelLabel)}${row.reference ? '<span class="ref-tag">REFERENCE</span>' : ''}</div></td>
            ${columns.map(([key]) => `<td class="metric-cell">${scoreCell(metric(row, key), state.view === 'ui' && state.uiMode === 'capture:vision' ? 'Not scored' : 'Not evaluated')}</td>`).join('')}
            <td><span class="status ${statusClass}">${statusLabel}</span></td><td><button class="row-open" data-result="${index}" aria-label="Open ${escapeHTML(row.model)} details">↗</button></td></tr>`;
        }).join('') || `<tr><td colspan="${columns.length + 3}" class="empty-state">No models match “${escapeHTML(state.query)}”.</td></tr>`}</tbody>
      </table>
    </div>
    <div class="table-note"><p>${note}</p>${methodLink()}</div>
    <span class="screen-reader" role="status">${filtered.length} results shown</span>`;
  renderHarnessComparison();
}

function methodLink() {
  const [id, label] = { ui: ['ui', 'UI protocol & scoring'], speed: ['scoring', 'Metric definitions'], archive: ['protocols', 'Protocol versions'] }[state.view] ?? ['tasks', 'Task selection & scoring'];
  return `<a href="methodology.html#${id}">${label} ↗</a>`;
}

function pageHero(title, description = '') {
  return `<section class="page-hero"><div class="container"><h1>${title}</h1>${description ? `<p class="lead">${description}</p>` : ''}</div></section>`;
}

const FLOWS = window.benchmarkArchitecture.suites;

function pipelinePage() {
  return `${pageHero('Evaluation pipeline')}<section class="container page-body architecture-page"><div class="tabs" role="tablist" aria-label="Evaluation suite">${Object.entries(FLOWS).map(([key, flow]) => `<button class="tab" id="suite-${key}" role="tab" aria-controls="pipeline-panel" data-suite="${key}" aria-selected="${state.suite === key}">${flow.label}${key === 'memory' ? ' <span class="subtle-tag">PLANNED</span>' : ''}</button>`).join('')}</div><div id="pipeline-panel" role="tabpanel" aria-labelledby="suite-${state.suite}"></div></section>`;
}

function renderFlow() {
  document.getElementById('pipeline-panel').innerHTML = window.benchmarkArchitecture.render(state.suite, state.component);
  window.benchmarkArchitecture.mount();
}

function taskSelectionSection() {
  const names = (setName, kind) => (DATA.taskSelection[setName] ?? []).filter(task => task.startsWith(`${kind}/`)).map(task => task.slice(kind.length + 1));
  const taskList = tasks => `<ul class="task-list">${tasks.map(task => `<li><code>${escapeHTML(task)}</code></li>`).join('')}</ul>`;
  return `<section class="article-section" id="tasks">
    <h2>Task selection</h2>
    <p>Fixed subsets of pinned Aider Python and MCPMark filesystem tasks, covering code editing and tool use.</p>
    <div class="results-table-wrap"><table class="selection-table"><thead><tr><th scope="col">Set</th><th scope="col">Source</th><th scope="col">Selection</th><th scope="col">Tasks / harness</th></tr></thead><tbody>
      <tr><th scope="row">Gate</th><td>Aider Python</td><td>5 short-solution core + 7 fixed extended exercises</td><td>12</td></tr>
      <tr><th scope="row">Gate</th><td>MCPMark filesystem</td><td>Registered easy tasks across 6 categories</td><td>10</td></tr>
      <tr><th scope="row">Ranking</th><td>Aider Python</td><td>All remaining exercises in the 34-task Python set</td><td>22</td></tr>
    </tbody></table></div>
    <p class="selection-explanation">The gate reuses the existing smoke suite to screen regressions. The ranking set is fixed by the pinned data, never chosen from the reference model’s failures. The two sets are disjoint and scored separately; each task runs once per harness, with an unscored warm-up.</p>
    <details class="task-details"><summary>View all 44 tasks and data versions</summary>
      <div class="task-list-group"><h3>Aider gate · 12 tasks</h3>${taskList(names('gate', 'aider'))}</div>
      <div class="task-list-group"><h3>MCPMark gate · 10 tasks</h3>${taskList(names('gate', 'mcpmark'))}</div>
      <div class="task-list-group"><h3>Aider ranking · 22 tasks</h3>${taskList(names('ranking', 'aider'))}</div>
      <p class="task-pins">Aider <code>7e0611e</code> · MCPMark <code>cd45b7f</code></p>
    </details>
    <div class="method-sources">${external('eval/backend/src/modeleval/capability/runner.py#L736', 'Selection code')} ${external('eval/agent_ability/smoke/tasks.py#L37', 'Gate registry')} ${external('eval/backend/pins.toml', 'Version pins')}</div>
  </section>`;
}

function uiMethodSection() {
  const row = DATA.rows.find(row => row.protocol === 'ui-real-1');
  if (!row) return '';
  const meta = row._ui_meta;
  return `<section class="article-section" id="ui"><h2>UI ability · ui-real-1</h2>
    <p>${meta.pages} captured public-web pages with offline replicas. The audit leaves ${meta.totalItems} items: ${meta.items.click} click targets, ${meta.items.fill} whole forms, and ${meta.items.read} reading questions, each sampled ${meta.samples} times. The task set is private; the audited counts and analysis are published.</p>
    <dl><div class="metric-definition"><dt>Inputs</dt><dd>Replica screenshot, screenshot + Apple Vision OCR, or an AX-style text list. Original captures are scored for reading only.</dd></div>
      <div class="metric-definition"><dt>Click / fill</dt><dd>Answers are replayed in WebKit and scored from page state. A form passes when every expected field is correct and protected fields remain unchanged.</dd></div>
      <div class="metric-definition"><dt>Reading</dt><dd>Strict uses the set’s exact text, number, or set matching. Lenient additionally normalizes typography and formatting, using the published scorer’s rules.</dd></div>
      <div class="metric-definition"><dt>Scope</dt><dd>A direct model probe with three correlated samples per item. It is separate from the three-harness capability task runs; latency is not reported because samples share prefix-cache hits.</dd></div>
    </dl><div class="method-sources">${external(row._details, 'Full protocol and failure analysis')}${external(row._ui_source, 'Audited counts')}${external('eval/agent_ability/ui_vision/real.py', 'Runner')}</div></section>`;
}

function observedSystems() {
  const groups = DATA.protocols.filter(group => group.machine === state.machine && group.kind !== 'archive');
  return `<strong>${escapeHTML(machineLabel())}</strong>${groups.map(group => {
    const row = DATA.rows.find(row => row.protocol === group.id && row.system.machine === group.machine);
    return `<div class="observed-system"><code>${escapeHTML(group.id)}</code> · ${escapeHTML(row.system.server)}<br>macOS ${escapeHTML(row.system.macos_build)} · Xcode ${escapeHTML(row.system.xcode_build)}</div>`;
  }).join('')}`;
}

function methodologyPage() {
  return `${pageHero('Methodology')}<section class="container page-body method-layout"><nav class="contents-nav" aria-label="On this page">${[['tasks','Task selection'],['ui','UI ability'],['scoring','Reading the scores'],['comparability','Fair comparisons'],['protocols','Protocol versions'],['limits','Limits & evidence']].map(([id,label]) => `<a class="${(location.hash.slice(1).split('/').pop() || 'tasks') === id ? 'selected' : ''}" href="methodology.html#${id}">${label}</a>`).join('')}</nav><article>${taskSelectionSection()}${uiMethodSection()}<section class="article-section" id="scoring"><h2>Reading the scores</h2><p>Task completion is a descriptive pass rate on the ranking set. It is reported separately from the gate set and tool-call checks.</p><dl><div class="metric-definition"><dt>Task completion</dt><dd>22 ranking tasks × 3 harnesses = 66 task runs per complete model result. A score of 63/66 is shown as 95.5%. Each task is run once per harness.</dd></div><div class="metric-definition"><dt>Gate set</dt><dd>12 Aider exercises + 10 MCPMark tasks, run through each of three harnesses. The gate determines whether to proceed to the ranking set.</dd></div><div class="metric-definition"><dt>Model time · median</dt><dd>For each harness, the median of summed model-call seconds per valid ranking task, including failed tasks.</dd></div><div class="metric-definition"><dt>Model calls · median</dt><dd>For the same ranking-task cohort, the median recorded <code>n_llm</code> count per task. Warm-ups and invalid records are excluded.</dd></div><div class="metric-definition"><dt>Tool use</dt><dd>Pass rate across the 22 M7 tool-call cases, separate from end-to-end task completion.</dd></div><div class="metric-definition"><dt>Cached TTFT</dt><dd>Time to first token using a cached ~15K-token agent prefix. Lower is better. Cold 64K TTFT is available in model details.</dd></div><div class="metric-definition"><dt>Decode @ 64K</dt><dd>Generation throughput with a 64K-token context, in tokens per second. Higher is better. The full run also reports 16K, 32K, and 110K contexts.</dd></div></dl><div class="method-callout">Ordering is descriptive. Sign-test decisions use task-level scores at p &lt; 0.05; three harness runs of the same task are correlated.</div><p>The gate calls a model worse only when it loses at least five more tasks than it wins, consistently in at least two harnesses. ${external('eval/backend/DESIGN.md', 'Read the decision rules')}</p></section><section class="article-section" id="comparability"><h2>Fair comparisons</h2><p>Compare results only within the same machine and protocol. The protocol fixes tasks, budgets, sampling rule, thinking setting, and server settings. Sampling follows each model’s recommended settings; it is a fixed rule, not identical numerical parameters for every model.</p><p>The main capability budget is 30 agent steps, 40 harness turns, and 3,600 seconds per task. Models run through Claude Code, Hermes Agent, and localmelo with identical task prompts, while retaining each harness’s own agent loop.</p><div class="method-callout">${observedSystems()}</div></section><section class="article-section" id="protocols"><h2>Protocol versions</h2><div class="results-table-wrap"><table class="protocol-table"><thead><tr><th scope="col">Protocol</th><th scope="col">Measures</th><th scope="col">State</th></tr></thead><tbody>${[...new Map(DATA.protocols.map(group => [group.id, group])).values()].map(group => `<tr><td><code>${escapeHTML(group.id)}</code></td><td>${escapeHTML(VIEW_LABELS[group.kind])}</td><td>${group.kind === 'archive' ? 'Historical' : 'Published'}</td></tr>`).join('')}</tbody></table></div><p style="margin-top:15px">Capability-2 uses harness epoch <code>4271c2f4a925</code> and serving protocol 2: a 12 GB hot cache and custom memory guard with a 90 GB ceiling. Flash-Next is the reference; numbers do not compare across protocols.</p><p>Historical protocols used a patched oMLX build. They are records of earlier experiments and cannot be rerun from this repository alone. Evolution ability (memory) has no published results.</p></section><section class="article-section" id="limits"><h2>Limits & evidence</h2><ul><li><strong>Partial runs:</strong> show observed counts and remain unranked. Nanbeige’s capability run was interrupted after 40 of 66 gate runs; its ranking set was not run.</li><li><strong>Timing disturbances:</strong> major disturbances trigger retries, up to three attempts. Minor disturbances are retained and flagged in model details.</li><li><strong>Small sample:</strong> Capability protocols have one run per task and harness; UI has three samples per item; the homepage displays no invented confidence intervals.</li><li><strong>Published evidence:</strong> summaries, manifests, speed records, and allowlisted capability metrics and checker output.</li><li><strong>Private evidence:</strong> capability request/response bodies, agent transcripts, harness homes, and server logs stay local.</li></ul>${external('docs/benchmark/README.md', 'Published data schema', false, 'text-link')}</section></article></section>`;
}

const COMMANDS = {
  setup: 'git clone https://github.com/xthomaswang/personal_agent_eval_pipeline.git\ncd personal_agent_eval_pipeline/eval/backend\n# Set machine-specific paths in config.toml.\nuv sync\nuv sync --project envs/capability\nuv sync --project envs/agent-tools\nuv run modeleval selftest',
  model: '# Add the model to models.toml first.\n# For a model with a prepare entry:\nuv run modeleval download <model>\nuv run modeleval convert <model>\nuv run modeleval register <model>\n# Build speed fixtures once, after preparing the reference.\n(cd harness && uv run agentbench fixtures build)',
  run: 'uv run modeleval plan <model>\nuv run modeleval run <model>\nuv run modeleval publish\ncd ../..\npython3 scripts/build_docs.py'
};

function codeBlock(key) {
  return `<div class="code-block"><button class="copy-button" data-copy="${key}">Copy</button><pre><code>${escapeHTML(COMMANDS[key])}</code></pre></div>`;
}

function reproducePage() {
  return `${pageHero('Reproduce', 'Prepare the reference model first, then evaluate new builds on your Mac.')}<section class="container page-body reproduce-grid"><div><section class="setup-step"><span class="step-badge">1</span><div><h2>Prepare the environment</h2><p>Configure local checkout paths, install the project environments, then use the offline self-test to find missing dependencies.</p>${codeBlock('setup')}</div></section><section class="setup-step"><span class="step-badge">2</span><div><h2>Register a model</h2><p>Use its prepare entry or register an existing checkpoint. On a new machine, prepare Qwen3.8-27B oQ5 + MTP as the reference first.</p>${codeBlock('model')}<div class="note-panel"><strong>The reference needs manual preparation.</strong> It has no prepare entry. Download the revision pinned in models.toml, convert with oq_convert.py, then register it. The fixture build reads its tokenizer: set <code>AGENTBENCH_TOKENIZER_DIR</code> to the checkpoint unless it is under <code>~/Developer/mlsys/models/oq/</code>. ${external('eval/backend/README.md', 'Read the full setup')}</div></div></section><section class="setup-step"><span class="step-badge">3</span><div><h2>Plan, evaluate, publish</h2><p>A run starts oMLX with fixed settings, measures serving performance, and evaluates agent ability. Publishing writes validated result rows; the site builder reads those rows.</p>${codeBlock('run')}<div class="note-panel"><strong>New-machine setup:</strong> verify and pin your harness epoch in pins.toml. A different macOS build, Xcode build, or epoch also needs a new protocol ID in publish.py before rows can be published.</div></div></section><div class="reproduce-source"><p>UI probes require a task set with replica pages and a scorer. ${external('eval/agent_ability/ui_vision/README.md', 'UI probe instructions')}</p>This is a quick guide. ${external('eval/backend/README.md', 'Full backend documentation')} · ${external('eval/backend/pins.toml', 'Version pins')}</div></div><aside class="prerequisites" aria-label="Prerequisites"><h3>Before you start</h3><div class="requirement"><b>Apple-silicon Mac</b><span>Record the machine and OS builds</span></div><div class="requirement"><b>uv</b><span>Project-managed Python environments</span></div><div class="requirement"><b>oMLX 0.7.1rc1</b><span>Commit 7a5a3d75 · its own .venv</span></div><div class="requirement"><b>Three agent harnesses</b><span>Claude Code · Hermes · localmelo</span></div><div class="requirement"><b>Pinned task data</b><span>Aider exercises + MCPMark fixtures</span></div><div class="requirement">${external('eval/backend/pins.toml', 'Check all pinned versions')}</div></aside></section>`;
}

function uiDetails(row) {
  const cell = pair => pair ? `${pair[0]}/${pair[1]} (${(100 * pair[0] / pair[1]).toFixed(1)}%)` : '—';
  return `<div class="results-table-wrap"><table class="protocol-table ui-detail-table"><thead><tr><th scope="col">Input</th><th scope="col">Click</th><th scope="col">Fill</th><th scope="col">Read · strict</th><th scope="col">Read · lenient</th></tr></thead><tbody>${Object.entries(UI_INPUTS).map(([arm, label]) => {
    const counts = row._ui[arm] ?? {};
    return `<tr><th scope="row">${label}</th><td>${cell(counts.click)}</td><td>${cell(counts.fill)}</td><td>${cell(counts.read_strict)}</td><td>${cell(counts.read_lenient)}</td></tr>`;
  }).join('')}</tbody></table></div>`;
}

function showResult(index) {
  const selected = DATA.rows[index];
  if (!selected) return;
  const related = DATA.rows.filter(row => row.family === selected.family && row.quant === selected.quant && row.run_id === selected.run_id && row.system.machine === selected.system.machine);
  document.getElementById('dialog-content').innerHTML = `<div class="eyebrow">Published result / ${selected.date}</div><h2 id="dialog-title">${escapeHTML(selected.model)}</h2><p class="quiet-meta">${escapeHTML(selected.system.machine)} · ${escapeHTML(selected.run_id)}</p><div class="detail-metrics">${related.map(row => `<h3>${row._ui ? 'UI ability' : row.suite === 'backend' ? 'Serving performance' : 'Agent ability'} <span class="ref-tag">${escapeHTML(row.protocol)}</span></h3>${row._ui ? uiDetails(row) : row.metrics.map(item => `<div class="detail-metric-row"><span>${escapeHTML(item.name)}</span><strong>${escapeHTML(item.value)}</strong></div>`).join('')}${row.notes ? `<div class="note-panel">${escapeHTML(row.notes)}</div>` : ''}<div class="dialog-source-links">${external(`${row._dir}/summary.json`, `${row.suite === 'backend' ? 'Speed' : 'Ability'} summary.json`)}${external(row._dir, 'Run evidence', true)}${row._details ? external(row._details, 'Full report') : ''}${row._ui_source ? external(row._ui_source, 'Audited counts') : ''}</div>`).join('')}</div><div class="dialog-provenance">macOS ${escapeHTML(selected.system.macos_build)} · Xcode ${escapeHTML(selected.system.xcode_build)}<br>${escapeHTML(selected.system.server)}<br>Dataset digest ${DATA.sourceDigest.slice(0, 12)}</div>`;
  dialog.showModal();
}

function resetSort() {
  state.sort = state.view === 'ui' && state.uiMode === 'capture:vision' ? 'UI read' : tableColumns()[0]?.[0] ?? '';
  state.ascending = /TTFT|memory|KL,/.test(state.sort);
}

function bindSearch() {
  document.getElementById('model-search')?.addEventListener('input', event => { state.query = event.target.value; renderResults(); });
}

function route() {
  const [fragment, detail] = location.hash.slice(1).split('/');
  const page = document.body.dataset.page || 'benchmark';
  const legacy = { pipeline: 'pipeline.html', methodology: 'methodology.html', reproduce: 'reproduce.html', how: 'pipeline.html', suites: 'pipeline.html', run: 'reproduce.html' };
  if (page === 'benchmark' && legacy[fragment]) {
    location.replace(`${legacy[fragment]}${detail ? `#${encodeURIComponent(detail)}` : ''}`);
    return;
  }
  const section = detail || fragment;
  if (page === 'benchmark' && Object.hasOwn(VIEW_LABELS, section)) { state.view = section; resetSort(); }
  document.querySelectorAll('[data-nav]').forEach(link => {
    if (link.dataset.nav === page) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  if (page === 'pipeline' && FLOWS[section]) { state.suite = section; state.component = null; }
  main.innerHTML = page === 'pipeline' ? pipelinePage() : page === 'methodology' ? methodologyPage() : page === 'reproduce' ? reproducePage() : benchmarkPage();
  if (page === 'benchmark') { resetSort(); renderResults(); bindSearch(); }
  if (page === 'pipeline') renderFlow();
  const anchor = page === 'methodology' && section ? document.getElementById(section)
    : page === 'benchmark' && section === 'harnesses' ? document.getElementById('harness-heading')
    : page === 'benchmark' && (Object.hasOwn(VIEW_LABELS, section) || section === 'results') ? document.getElementById('results-heading') : null;
  if (anchor) anchor.scrollIntoView({ behavior: 'instant', block: 'start' });
  else window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
}

main.addEventListener('click', async event => {
  const view = event.target.closest('[data-view]');
  if (view) {
    state.view = view.dataset.view; state.query = ''; resetSort();
    history.replaceState(null, '', `#${state.view}`);
    main.innerHTML = benchmarkPage(); renderResults();
    bindSearch();
    document.getElementById(`tab-${state.view}`).focus({ preventScroll: true });
  }
  const sort = event.target.closest('[data-sort]');
  if (sort) { state.ascending = state.sort === sort.dataset.sort ? !state.ascending : /TTFT|memory|KL,|Median/.test(sort.dataset.sort); state.sort = sort.dataset.sort; renderResults(); }
  const result = event.target.closest('[data-result]');
  if (result) showResult(Number(result.dataset.result));
  const suite = event.target.closest('[data-suite]');
  if (suite) { location.hash = suite.dataset.suite; }
  const component = event.target.closest('[data-component]');
  if (component) { state.component = component.dataset.component; renderFlow(); document.querySelector(`[data-component="${state.component}"]`).focus({ preventScroll: true }); }
  const copy = event.target.closest('[data-copy]');
  if (copy) {
    try { await navigator.clipboard.writeText(COMMANDS[copy.dataset.copy]); copy.textContent = 'Copied'; }
    catch { copy.textContent = 'Select text'; }
  }
});


main.addEventListener('change', event => {
  const id = event.target.id;
  if (id === 'comparison-model') { state.compareModel = event.target.value; renderHarnessComparison(); return; }
  if (id === 'result-machine') {
    state.machine = event.target.value; state.query = ''; state.compareModel = null; state.selectedProtocols = {};
    main.innerHTML = benchmarkPage(); resetSort(); renderResults(); bindSearch(); return;
  }
  if (id === 'result-protocol') {
    state.selectedProtocols[state.view] = event.target.value; state.compareModel = null;
    main.innerHTML = benchmarkPage(); resetSort(); renderResults(); bindSearch();
    document.getElementById(id)?.focus({ preventScroll: true }); return;
  }
  else if (id === 'ui-input') state.uiMode = event.target.value;
  else if (id === 'ui-scoring') state.readScoring = event.target.value;
  else return;
  resetSort(); renderResults();
  document.getElementById(id)?.focus({ preventScroll: true });
});

main.addEventListener('keydown', event => {
  const tab = event.target.closest('[role="tab"]');
  if (!tab || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const tabs = [...tab.parentElement.querySelectorAll('[role="tab"]')];
  const index = tabs.indexOf(tab);
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus(); tabs[next].click();
});

document.querySelector('.skip-link').addEventListener('click', event => {
  event.preventDefault();
  main.focus();
  main.scrollIntoView();
});
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
window.addEventListener('hashchange', route);
route();
