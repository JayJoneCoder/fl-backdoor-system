import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compile = path => ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

const createHarness = () => {
  const slots = [], effects = [], charts = [], sockets = [], progress = [];
  const frames = new Map(), timers = new Map();
  let cursor = 0, tree;
  const react = {
    useState: initial => {
      const index = cursor++;
      const slot = slots[index] ??= { value: initial };
      return [slot.value, update => { slot.value = typeof update === 'function' ? update(slot.value) : update; }];
    },
    useRef: initial => slots[cursor++] ??= { current: initial },
    useEffect: (callback, deps) => {
      const index = cursor++;
      if (!slots[index] || deps.some((value, i) => value !== slots[index].deps[i])) {
        const old = slots[index];
        const slot = slots[index] = { deps };
        effects.push(() => { old?.cleanup?.(); slot.cleanup = callback(); });
      }
    },
  };
  const jsx = (type, props) => ({ type, props });
  class Socket {
    constructor(url) { this.url = url; sockets.push(this); }
    receive(data) { this.onmessage?.({ data: typeof data === 'string' ? data : JSON.stringify(data) }); }
    disconnect() { this.onclose?.(); }
    close() { this.closed = true; }
  }
  const load = (path, imports = {}, globals = {}) => {
    const exports = {};
    runInNewContext(compile(path), { exports, ...globals, require: name => {
      assert.ok(name in imports, name); return imports[name];
    } });
    return exports;
  };
  const utils = load('../src/utils/experimentStream.ts');
  const component = load('../src/components/BatchMonitor/ExperimentMetrics.tsx', {
    react, 'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' },
    antd: { Alert: 'Alert', Button: 'Button', Card: 'Card', Col: 'Col', Row: 'Row', Typography: { Text: 'Text', Paragraph: 'Paragraph' } },
    echarts: { init: () => {
      const chart = { options: {}, disposed: false,
        setOption(value) { this.options = { ...this.options, ...JSON.parse(JSON.stringify(value)) }; },
        resize() {}, dispose() { this.disposed = true; } };
      charts.push(chart); return chart;
    } },
    '../../api/endpoints': { getWebSocketUrl: path => path },
    '../../api/checkedFetch': load('../src/api/checkedFetch.ts'),
    '../../utils/experimentStream': utils,
  }, {
    WebSocket: Socket,
    window: { addEventListener() {}, removeEventListener() {} },
    requestAnimationFrame: callback => { const id = Symbol(); frames.set(id, callback); return id; },
    cancelAnimationFrame: id => frames.delete(id),
    setTimeout: callback => { const id = Symbol(); timers.set(id, callback); return id; },
    clearTimeout: id => timers.delete(id),
  }).default;
  const visit = (node, callback) => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { node.forEach(child => visit(child, callback)); return; }
    callback(node); visit(node.props?.children, callback);
  };
  const onProgress = value => progress.push(JSON.parse(JSON.stringify(value)));
  const render = () => {
    cursor = 0;
    tree = component({ name: 'experiment one', runId: 'run/one', onProgress });
    visit(tree, node => { if (node.props?.ref) node.props.ref.current ??= {}; });
    effects.splice(0).forEach(callback => callback());
  };
  const flush = () => { const pending = [...frames.values()]; frames.clear(); pending.forEach(callback => callback()); render(); };
  render();
  return { sockets, charts, progress, flush,
    reconnect: () => { const pending = [...timers.values()]; timers.clear(); pending.forEach(callback => callback()); render(); },
    alerts: () => { const found = []; visit(tree, node => { if (node.type === 'Alert') found.push(node); }); return found; },
    paragraphs: () => { const found = []; visit(tree, node => { if (node.type === 'Paragraph') found.push(node.props.children); }); return found; },
    receive: packet => { sockets.at(-1).receive(packet); render(); },
    unmount: () => slots.forEach(slot => slot?.cleanup?.()),
  };
};

const point = (round, accuracy = 0.8) => ({ type: 'csv_update', data: { round, accuracy, asr: 0.2, loss: 1.2 } });

test('批量图表复用 CSV 指标格式，重放乱序去重，轮次与百分比来自实际指标', () => {
  const h = createHarness();
  try {
    assert.match(h.sockets[0].url, /experiment%20one\?run_id=run%2Fone/);
    h.receive({ type: 'status', data: { status: 'running', total_rounds: 10 } });
    assert.equal(h.progress.at(-1).percent, 0);
    h.receive(point(4)); h.receive(point(2)); h.receive(point(4, 0.9));
    h.receive('[ROUND 5/10] plain log is not a metric');
    h.receive({ type: 'csv_update', data: { round: 8, accuracy: 'invalid' } });
    h.flush();
    assert.deepEqual(h.charts[0].options.xAxis.data, [2, 4]);
    assert.deepEqual(h.charts[0].options.series[0].data, [0.8, 0.9]);
    assert.deepEqual(h.charts[1].options.series[0].data, [1.2, 1.2]);
    assert.equal(h.progress.at(-1).percent, 40);
    assert.equal(h.progress.at(-1).runId, 'run/one');
  } finally { h.unmount(); }
});

test('指标断线自动重连且不重复曲线，卸载后不接受旧消息并释放图表', () => {
  const h = createHarness();
  h.receive(point(1)); h.flush();
  h.sockets[0].disconnect(); h.reconnect();
  assert.equal(h.sockets.length, 2);
  h.receive(point(1, 0.85)); h.receive(point(2)); h.flush();
  assert.deepEqual(h.charts[0].options.xAxis.data, [1, 2]);
  assert.deepEqual(h.charts[0].options.series[0].data, [0.85, 0.8]);
  h.unmount();
  const count = h.progress.length;
  h.sockets[1].receive(point(8));
  assert.equal(h.progress.length, count);
  assert.ok(h.charts.every(chart => chart.disposed));
  assert.equal(h.sockets[1].closed, true);
});

test('指标流失败原因显示简明中文诊断', () => {
  const h = createHarness();
  try {
    h.receive({ type: 'status', data: { status: 'failed', error: 'Experiment worker exited with code 7. See run.log for details.' } });
    assert.match(h.alerts()[0].props.title, /实验运行失败（退出码 7）/);
  } finally { h.unmount(); }
});

test('图例显式下移且为横轴名称、刻度和图例预留分离的底部区域', () => {
  const h = createHarness();
  try {
    for (const chart of h.charts) {
      assert.equal(chart.options.legend.bottom, 4);
      assert.ok(chart.options.grid.bottom >= 80);
      assert.equal(chart.options.xAxis.nameLocation, 'middle');
    }
    assert.equal(h.charts[0].options.legend.data[0], '准确率 (ACC)');
    assert.equal(h.charts[1].options.legend.data[0], '损失 (Loss)');
  } finally { h.unmount(); }
});

test('终态后仍接收最后一轮指标，正常断线不重连也不误报连接错误', () => {
  for (const state of ['finished', 'failed', 'cancelled']) {
    const h = createHarness();
    try {
      h.receive({ type: 'status', data: { status: state, total_rounds: 10 } });
      assert.match(h.paragraphs()[0], /已结束，未生成可用指标/);
      h.receive(point(10)); h.flush();
      assert.deepEqual(h.charts[0].options.xAxis.data, [10]);
      h.sockets[0].disconnect(); h.reconnect();
      assert.equal(h.sockets.length, 1);
      assert.equal(h.alerts().length, 0);
      assert.equal(h.paragraphs().length, 0);
    } finally { h.unmount(); }
  }
});