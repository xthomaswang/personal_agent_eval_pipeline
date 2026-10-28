'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    id, innerHTML: '', handlers: {}, addEventListener(type, fn) { this.handlers[type] = fn; },
    focus() {}, scrollIntoView() {}, setAttribute() {}, removeAttribute() {},
  });
  return elements.get(id);
}
const row = (protocol, model, reference) => ({
  protocol, model, reference, quant: 'oq5', date: '2026-10-28', _dir: protocol, system: { machine: 'mac' },
  metrics: [{ name: 'Ranking set', value: '60/66' }, { name: 'Gate set', value: '60/66' },
    { name: 'M7 tool-call cases', value: '—' }],
  _harnesses: {}, _harness_source: 'results.json',
});
const data = {
  repoUrl: 'https://example.org', machines: [{ id: 'mac', label: 'Mac' }], latest: '2026-10-28',
  protocols: ['capability-1', 'capability-2'].map(id => ({ machine: 'mac', kind: 'ability', id, latest: '2026-10-28' })),
  rows: [row('capability-1', 'Old reference', true), row('capability-2', 'New reference', true)],
};
const document = {
  getElementById: element, querySelector: () => element('skip-link'), querySelectorAll: () => [],
  body: { dataset: { page: 'benchmark' } },
};
const context = vm.createContext({
  assert, document, window: { BENCHMARK_DATA: data, benchmarkArchitecture: { suites: {} }, addEventListener() {} },
  location: { hash: '' }, history: { replaceState() {} }, navigator: {}, console,
});
const source = fs.readFileSync(process.argv[2], 'utf8').replace(/\nroute\(\);\s*$/, '\n');
vm.runInContext(source + `
  assert.equal(protocol(), 'capability-1');
  assert.equal(capabilityRows()[0].model, 'Old reference');
  assert.match(benchmarkPage(), /capability-1 ·/);
  assert.equal(tableColumns()[0][1], 'Task completion');
  main.handlers.change({ target: { id: 'result-protocol', value: 'capability-2' } });
  assert.equal(protocol(), 'capability-2');
  assert.equal(capabilityRows().length, 1);
  assert.equal(comparisonModel().model, 'New reference');
  assert.match(main.innerHTML, /capability-2 ·/);
  assert.doesNotMatch(main.innerHTML, /Old reference/);
  assert.equal(tableColumns()[0][1], 'Task completion');
  assert.match(document.getElementById('table-results').innerHTML, /22 ranking tasks/);
  main.handlers.change({ target: { id: 'result-protocol', value: 'capability-1' } });
  assert.equal(comparisonModel().model, 'Old reference');
  assert.match(main.innerHTML, /capability-1 ·/);
`, context);
