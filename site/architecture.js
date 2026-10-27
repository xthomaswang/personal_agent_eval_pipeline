'use strict';

// Editable component diagram, with connectors measured from the rendered nodes.
window.benchmarkArchitecture = (() => {
  const suites = {
    capability: { label: 'Agent ability', protocol: 'capability-1', selected: 'harnesses' },
    ui: { label: 'UI ability', protocol: 'ui-real-1', selected: 'ui' },
    speed: { label: 'Serving performance', protocol: 'speed-1', selected: 'speed' },
    memory: { label: 'Evolution ability (memory)', protocol: 'In development', selected: 'memory' }
  };
  const components = {
    ui: {
      title: 'UI ability', subtitle: 'Screenshot · OCR · AX',
      description: 'The real-web probe evaluates clicking, whole-form filling, and reading on captured public-web pages and offline replicas. Only audited counts and analysis are published.',
      source: 'eval/agent_ability/ui_vision/README.md'
    },
    probe: {
      title: 'Model probe', subtitle: 'Direct requests · 3 samples',
      description: 'Each item is sampled three times with a screenshot, screenshot plus OCR, or an accessibility-style list. Original captures are tested on reading only. Requests go directly to the model server.',
      source: 'eval/agent_ability/ui_vision/real.py'
    },
    replay: {
      title: 'WebKit replay', subtitle: 'Click · fill · page state',
      description: 'Click and fill answers are replayed on offline replica pages. The scorer checks what the page recorded, including whether protected form fields stayed unchanged.',
      source: 'eval/agent_ability/ui_vision/webrun.swift'
    },
    speed: {
      title: 'Serving performance', subtitle: 'TTFT · decode · memory',
      description: 'Fixed prompts measure latency, decoding at 16K–110K context, a 20-step trajectory, and peak server memory. Timed probes call oMLX directly.',
      source: 'eval/backend/README.md'
    },
    tasks: {
      title: 'Agent ability', subtitle: 'Aider + MCPMark',
      description: 'The gate has 12 Aider exercises and 10 MCPMark tasks. The ranking set has 22 more Aider exercises. Each runs through all three harnesses.',
      source: 'eval/agent_ability/README.md'
    },
    memory: {
      title: 'Evolution ability', subtitle: 'Memory · 10-day timeline', planned: true,
      description: '86 synthetic entries over 10 days, followed by 60 memory-only questions. Fixtures, reader, and oracle exist; the daily session runner is pending.',
      source: 'eval/memory/README.md'
    },
    harnesses: {
      title: 'Agent harnesses', subtitle: 'Claude Code · Hermes · localmelo',
      description: 'Identical task prompts run through three independent agent loops, each with a budget of 30 steps, 40 turns, and 3,600 seconds.',
      source: 'eval/bench/harness.py'
    },
    tools: {
      title: 'Tools & workspace', subtitle: 'Built-in tools · task fixtures',
      description: 'Each harness uses its built-in tools in the task workspace. Objective checkers run outside that workspace; Aider tests are restored before scoring.',
      source: 'eval/agent_ability/README.md'
    },
    recorder: {
      title: 'Recorder proxy', subtitle: 'Sampling · steps · model time',
      description: 'Every harness model call passes through the shared recorder. It applies the sampling rule and thinking setting, then records steps and model time.',
      source: 'eval/bench/proxy', tree: true
    },
    server: {
      title: 'oMLX server', subtitle: 'Local model · Apple silicon',
      description: 'One model runs on an Apple-silicon Mac with versioned server settings. Speed probes bypass the recorder; harness requests pass through it.',
      source: 'eval/backend/README.md'
    },
    checkers: {
      title: 'Objective checkers', subtitle: 'Tests · verifiers · oracle',
      description: 'Aider unit tests and MCPMark verifiers score task completion. The evolution suite’s reader and oracle are built, but agent-memory results are not yet available.',
      source: 'eval/agent_ability/README.md'
    },
    results: {
      title: 'Published results', subtitle: 'summary.json · run evidence',
      description: 'modeleval publish writes validated rows and allowlisted evidence. Each result records its machine, protocol, server version, and macOS and Xcode builds.',
      source: 'docs/benchmark/README.md'
    },
    site: {
      title: 'Benchmark website', subtitle: 'Generated from published rows',
      description: 'The site builder groups results by machine and protocol. Capability, speed, and historical comparisons stay separate.',
      source: 'scripts/build_docs.py'
    }
  };
  const routes = {
    capability: ['tasks', 'harnesses', 'tools', 'recorder', 'server', 'checkers', 'results', 'site'],
    ui: ['ui', 'probe', 'replay', 'server', 'checkers', 'results', 'site'],
    speed: ['speed', 'server', 'results', 'site'],
    memory: ['memory', 'harnesses', 'tools', 'recorder', 'server', 'checkers', 'results', 'site']
  };
  const connections = [
    { from: 'speed', to: 'server', route: 'direct', suite: 'speed', label: 'direct probes' },
    { from: 'tasks', to: 'harnesses', route: 'across', suite: 'capability' },
    { from: 'memory', to: 'harnesses', route: 'across', suite: 'memory', planned: true },
    { from: 'harnesses', to: 'tools', route: 'down', suite: 'agent', twoWay: true },
    { from: 'harnesses', to: 'recorder', route: 'across', suite: 'agent', label: 'model calls' },
    { from: 'recorder', to: 'server', route: 'down', suite: 'agent' },
    { from: 'tools', to: 'checkers', route: 'score', suite: 'agent' },
    { from: 'checkers', to: 'results', route: 'across', suite: 'scored' },
    { from: 'server', to: 'results', route: 'records', suite: 'speed', label: 'timings' },
    { from: 'results', to: 'site', route: 'across', suite: 'all' },
    { from: 'ui', to: 'probe', route: 'across', suite: 'ui' },
    { from: 'probe', to: 'server', route: 'across', suite: 'ui', twoWay: true, label: 'model calls' },
    { from: 'probe', to: 'replay', route: 'down', suite: 'ui' },
    { from: 'replay', to: 'checkers', route: 'score', suite: 'ui' }
  ];
  let observer;
  let frame;

  function render(suite, selection) {
    const chosen = suites[suite];
    const selected = components[selection] ? selection : chosen.selected;
    const node = key => {
      const c = components[key];
      const active = routes[suite].includes(key);
      return `<button class="arch-node ${c.planned ? 'is-planned' : ''} ${active ? 'is-active' : ''}" id="arch-${key}" data-component="${key}" aria-pressed="${selected === key}" aria-controls="component-detail"><strong>${c.title}</strong><span>${c.subtitle}</span>${c.planned ? '<small>PLANNED</small>' : ''}</button>`;
    };
    const c = components[selected];
    return `<div class="architecture-heading"><h2>Architecture</h2><span class="quiet-meta">${chosen.protocol}</span></div>
      <div class="architecture-scroll"><div class="architecture-map" id="architecture-map" data-suite="${suite}" aria-label="Evaluation architecture: suites, execution, local inference, scoring, and publication. Selected route: ${chosen.label}.">
        <div class="arch-group arch-suites"><span class="arch-group-label">Evaluation suites</span>${node('speed')}${node('tasks')}${node('ui')}${node('memory')}</div>
        <div class="arch-group arch-execution"><span class="arch-group-label">${suite === 'ui' ? 'UI evaluation' : 'Agent execution'}</span>${suite === 'ui' ? node('probe') + node('replay') : node('harnesses') + node('tools')}</div>
        <div class="arch-group arch-serving"><span class="arch-group-label">Local inference</span>${node('recorder')}${node('server')}</div>
        <div class="arch-group arch-publication"><span class="arch-group-label">Scoring & publication</span>${node('checkers')}${node('results')}${node('site')}</div>
        <svg class="architecture-connections" id="architecture-connections" aria-hidden="true"><defs><marker id="arch-arrow" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 4L0 8Z" fill="context-stroke"/></marker></defs><g id="architecture-edges"></g></svg>
      </div></div>
      <div class="architecture-legend"><span><i></i>Selected suite</span><span><i class="dashed"></i>Planned path</span><span class="legend-hint">Select a component for details</span></div>
      <div class="component-detail" id="component-detail" aria-live="polite"><h3>${c.title}${selected === 'memory' ? ' (memory)' : ''}</h3><p>${c.description}</p><a class="code-link" href="${window.BENCHMARK_DATA.repoUrl}/${c.tree ? 'tree' : 'blob'}/main/${c.source}" target="_blank" rel="noreferrer">${c.source} ↗</a></div>`;
  }

  function draw() {
    const map = document.getElementById('architecture-map');
    if (!map) return;
    const edges = document.getElementById('architecture-edges');
    const svg = document.getElementById('architecture-connections');
    const bounds = map.getBoundingClientRect();
    const mobile = bounds.width < 800;
    const suite = map.dataset.suite;
    svg.setAttribute('viewBox', `0 0 ${bounds.width} ${bounds.height}`);
    const rect = key => {
      const r = document.getElementById(`arch-${key}`).getBoundingClientRect();
      return { x: r.left - bounds.left, y: r.top - bounds.top, w: r.width, h: r.height };
    };
    edges.replaceChildren();
    connections.forEach(edge => {
      if (!document.getElementById(`arch-${edge.from}`) || !document.getElementById(`arch-${edge.to}`)) return;
      const a = rect(edge.from), b = rect(edge.to);
      const active = edge.suite === suite || edge.suite === 'all'
        || edge.suite === 'agent' && ['capability', 'memory'].includes(suite)
        || edge.suite === 'scored' && ['capability', 'memory', 'ui'].includes(suite);
      let points;
      let label;
      if (edge.route === 'direct') {
        const x = bounds.width - 8;
        if (mobile) {
          const y = a.y - 12;
          points = [[a.x + a.w / 2, a.y], [a.x + a.w / 2, y], [x, y], [x, b.y + b.h / 2], [b.x + b.w, b.y + b.h / 2]];
          label = [x - 4, (a.y + b.y) / 2, 'end'];
        } else {
          const y = a.y + a.h / 2;
          points = [[a.x + a.w, y], [x, y], [x, b.y + b.h / 2], [b.x + b.w, b.y + b.h / 2]];
          label = [bounds.width * .61, y - 7, 'middle'];
        }
      } else if (edge.route === 'down') {
        points = [[a.x + a.w / 2, a.y + a.h], [b.x + b.w / 2, b.y]];
      } else if (edge.route === 'score') {
        // Cross in the gap above the publication group, so the line does not run through the suite nodes.
        const top = Math.max(...['.arch-suites', '.arch-execution', '.arch-serving']
          .map(selector => map.querySelector(selector).getBoundingClientRect().bottom)) - bounds.top;
        const y = (top + map.querySelector('.arch-publication').getBoundingClientRect().top - bounds.top) / 2;
        points = [[a.x + a.w * .35, a.y + a.h], [a.x + a.w * .35, y], [b.x + b.w / 2, y], [b.x + b.w / 2, b.y]];
      } else if (edge.route === 'records') {
        const y = b.y - 18;
        points = [[a.x + a.w * .7, a.y + a.h], [a.x + a.w * .7, y], [b.x + b.w / 2, y], [b.x + b.w / 2, b.y]];
        label = [a.x + a.w * .7 + 7, y - 6, 'start'];
      } else if (mobile && ['tasks', 'memory', 'ui'].includes(edge.from)) {
        const railY = Math.max(rect('tasks').y + rect('tasks').h, rect('memory').y + rect('memory').h) + 18 + (edge.from === 'memory' ? 7 : 0);
        points = [[a.x + a.w / 2, a.y + a.h], [a.x + a.w / 2, railY], [b.x + b.w * (edge.from === 'tasks' ? .4 : .6), railY], [b.x + b.w * (edge.from === 'tasks' ? .4 : .6), b.y]];
      } else {
        const x = a.x + a.w + (b.x - a.x - a.w) * .5;
        points = [[a.x + a.w, a.y + a.h / 2], [x, a.y + a.h / 2], [x, b.y + b.h / 2], [b.x, b.y + b.h / 2]];
        if (edge.label) label = [(a.x + a.w + b.x) / 2, a.y + a.h / 2 - 8, 'middle'];
      }
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', points.map((p, i) => `${i ? 'L' : 'M'}${p[0]},${p[1]}`).join(' '));
      path.setAttribute('class', `arch-edge ${active ? 'active' : ''} ${edge.planned || suite === 'memory' && edge.suite === 'agent' ? 'planned' : ''}`);
      path.setAttribute('marker-end', 'url(#arch-arrow)');
      if (edge.twoWay) path.setAttribute('marker-start', 'url(#arch-arrow)');
      edges.append(path);
      if (label && active) {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', label[0]); text.setAttribute('y', label[1]); text.setAttribute('text-anchor', label[2]);
        text.setAttribute('class', 'arch-edge-label'); text.textContent = edge.label;
        edges.append(text);
      }
    });
  }

  function mount() {
    const map = document.getElementById('architecture-map');
    observer?.disconnect();
    cancelAnimationFrame(frame);
    if (!map) return;
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => { if (map.isConnected) draw(); });
    };
    observer = new ResizeObserver(schedule);
    observer.observe(map);
    document.fonts.ready.then(schedule);
    schedule();
  }

  return { suites, render, mount };
})();
