import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

// Exercise the actual component and log hook using the project's TS compiler.
// Hooks, timers, ECharts and WebSocket are deterministic substitutes; no server,
// DOM package, training process or additional test framework is needed.
const compile = relativePath => ts.transpileModule(
  readFileSync(new URL(relativePath, import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const componentCode = compile('../src/components/ExperimentControl/index.tsx');
const hookCode = compile('../src/hooks/useBufferedLogs.ts');
const utilityCode = compile('../src/utils/experimentStream.ts');
const copy = value => JSON.parse(JSON.stringify(value));

const createHarness = (initialStatus = { status: 'running', exp_name: 'same_name', total_rounds: 10 }) => {
  const slots = [];
  const pendingEffects = [];
  const frames = new Map();
  const intervals = new Map();
  const charts = [];
  const sockets = [];
  let cursor = 0;
  let nextId = 1;
  let dirty = false;
  let tree;
  let currentStatus = initialStatus;
  const equalDeps = (left, right) => left && right && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
  const react = {
    useState: initial => {
      const index = cursor++;
      const slot = slots[index] ??= { value: initial };
      return [slot.value, update => {
        const value = typeof update === 'function' ? update(slot.value) : update;
        if (!Object.is(slot.value, value)) { slot.value = value; dirty = true; }
      }];
    },
    useRef: initial => {
      const index = cursor++;
      return slots[index] ??= { current: initial };
    },
    useCallback: (callback, deps) => {
      const index = cursor++;
      if (!slots[index] || !equalDeps(slots[index].deps, deps)) slots[index] = { callback, deps };
      return slots[index].callback;
    },
    useEffect: (callback, deps) => {
      const index = cursor++;
      if (!slots[index] || !equalDeps(slots[index].deps, deps)) {
        const previous = slots[index];
        const slot = { deps, cleanup: null };
        slots[index] = slot;
        pendingEffects.push(() => { previous?.cleanup?.(); slot.cleanup = callback(); });
      }
    },
  };
  const jsx = (type, props) => ({ type, props });
  const visit = (node, callback) => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { node.forEach(child => visit(child, callback)); return; }
    callback(node);
    visit(node.props?.children, callback);
  };
  class Socket {
    constructor(url) { this.url = url; this.closed = false; sockets.push(this); }
    close() { this.closed = true; }
    disconnect() { this.closed = true; this.onclose?.(); }
    receive(data) { this.onmessage?.({ data: typeof data === 'string' ? data : JSON.stringify(data) }); }
  }
  const echarts = {
    init: element => {
      const chart = {
        element, disposed: false, options: {},
        setOption(option) { this.options = { ...this.options, ...copy(option) }; },
        resize() {},
        dispose() { this.disposed = true; },
      };
      charts.push(chart);
      return chart;
    },
  };
  const api = {
    getExperimentStatus: async () => ({ data: { ...currentStatus } }),
    getConfig: async () => ({ data: { 'num-server-rounds': 10 } }),
    startExperiment: async () => {},
    stopExperiment: async () => {},
  };
  const globals = {
    requestAnimationFrame: callback => { const id = nextId++; frames.set(id, callback); return id; },
    cancelAnimationFrame: id => frames.delete(id),
    setInterval: (callback, duration) => { const id = nextId++; intervals.set(id, { callback, duration }); return id; },
    clearInterval: id => intervals.delete(id),
    setTimeout, clearTimeout,
    window: { addEventListener() {}, removeEventListener() {} },
    WebSocket: Socket,
  };
  const load = (code, imports) => {
    const exports = {};
    runInNewContext(code, {
      ...globals, exports,
      require: name => {
        if (!(name in imports)) throw new Error(`Unexpected import: ${name}`);
        return imports[name];
      },
    });
    return exports;
  };
  const utils = load(utilityCode, {});
  const hook = load(hookCode, { react, '../utils/experimentStream': utils });
  const antd = Object.fromEntries(['Button', 'Space', 'Input', 'Card', 'Tag', 'Alert', 'Progress'].map(name => [name, name]));
  antd.message = { success() {}, error() {} };
  const component = load(componentCode, {
    react,
    'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' },
    antd,
    '@ant-design/icons': { PlayCircleOutlined: 'Play', StopOutlined: 'Stop', SyncOutlined: 'Sync' },
    echarts,
    '../../api/client': api,
    '../../api/endpoints': { getWebSocketUrl: path => `ws://test${path}` },
    '../../components/LogViewer': { __esModule: true, default: 'LogViewer' },
    '../../hooks/useBufferedLogs': hook,
    '../../utils/experimentStream': utils,
  }).default;
  const render = () => {
    do {
      dirty = false;
      cursor = 0;
      tree = component({});
      visit(tree, node => {
        if (node.props?.ref && !node.props.ref.current) node.props.ref.current = {};
      });
      pendingEffects.splice(0).forEach(effect => effect());
    } while (dirty);
  };
  const settle = async () => {
    // Drain status polling's promise continuations, then React state updates.
    await Promise.resolve();
    await Promise.resolve();
    render();
  };
  const runFrames = () => {
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach(callback => callback());
    render();
  };
  const runLastFrame = () => {
    const last = [...frames.entries()].at(-1);
    if (!last) return;
    frames.delete(last[0]);
    last[1]();
    render();
  };
  const tick = async duration => {
    [...intervals.values()].filter(timer => timer.duration === duration).forEach(timer => timer.callback());
    await settle();
  };
  const find = type => {
    let match;
    visit(tree, node => { if (node.type === type) match = node; });
    return match;
  };
  render();
  return {
    sockets, charts, settle, runFrames, runLastFrame, tick,
    setStatus: status => { currentStatus = status; },
    logs: () => copy(find('LogViewer').props.logs),
    progress: () => find('Progress')?.props.percent,
    unmount: () => slots.forEach(slot => slot?.cleanup?.()),
    intervals: () => intervals.size,
    frames: () => frames.size,
  };
};

const point = round => ({ round, accuracy: round / 10, asr: 0.2, loss: 1.5 - round / 10 });
const sendPoint = (socket, round) => socket.receive({ type: 'csv_update', data: point(round) });
const sendStatus = (socket, status = 'running') => socket.receive({
  type: 'status', data: { status, exp_name: 'same_name', total_rounds: 10 },
});
const assertCharts = (harness, rounds) => {
  assert.equal(harness.charts.length, 2);
  assert.deepEqual(harness.charts[0].options.xAxis.data, rounds);
  assert.deepEqual(harness.charts[1].options.xAxis.data, rounds);
  assert.deepEqual(harness.charts[0].options.series[0].data, rounds.map(round => point(round).accuracy));
  assert.deepEqual(harness.charts[1].options.series[0].data, rounds.map(round => point(round).loss));
};

test('刷新时历史数据早于图表初始化也能完整绘制，无需等待下一轮', async () => {
  const harness = createHarness();
  try {
    await harness.settle();
    const socket = harness.sockets[0];
    assert.ok(socket);
    assert.equal(harness.charts.length, 0);
    sendStatus(socket);
    [0, 1, 2, 3].forEach(round => sendPoint(socket, round));
    socket.receive('starting');
    socket.receive('round three');
    await harness.tick(200);
    // Force a paint attempt before chart initialization; its data must survive.
    harness.runLastFrame();
    assert.equal(harness.charts.length, 0);
    harness.runFrames();
    assertCharts(harness, [0, 1, 2, 3]);
    assert.equal(harness.progress(), 30);
    assert.deepEqual(harness.logs(), ['starting', 'round three']);
  } finally { harness.unmount(); }
});

test('切页卸载清理连接和定时器，返回后恢复期间已完成的轮次与日志', async () => {
  const first = createHarness();
  await first.settle();
  const oldSocket = first.sockets[0];
  sendStatus(oldSocket);
  [0, 1].forEach(round => sendPoint(oldSocket, round));
  first.runFrames();
  first.unmount();
  assert.equal(oldSocket.closed, true);
  assert.equal(first.intervals(), 0);
  assert.equal(first.frames(), 0);
  assert.equal(first.charts.every(chart => chart.disposed), true);

  const returned = createHarness();
  try {
    returned.runFrames();
    await returned.settle();
    const socket = returned.sockets[0];
    sendStatus(socket);
    [0, 1, 2, 3].forEach(round => sendPoint(socket, round));
    ['starting', 'round one', 'round two', 'round three'].forEach(line => socket.receive(line));
    // Late messages from the disposed page cannot touch the new component.
    sendPoint(oldSocket, 9);
    oldSocket.receive('stale log');
    await returned.tick(200);
    returned.runFrames();
    assertCharts(returned, [0, 1, 2, 3]);
    assert.deepEqual(returned.logs(), ['starting', 'round one', 'round two', 'round three']);
    assert.equal(returned.progress(), 30);
  } finally { returned.unmount(); }
});

test('断线重连完整回放替换旧数据，日志不重复，迟到的旧连接消息被忽略', async () => {
  const harness = createHarness();
  try {
    await harness.settle();
    const oldSocket = harness.sockets[0];
    sendStatus(oldSocket);
    [0, 1].forEach(round => sendPoint(oldSocket, round));
    oldSocket.receive('starting');
    await harness.tick(200);
    harness.runFrames();
    oldSocket.disconnect();
    await harness.tick(5000);
    assert.equal(harness.sockets.length, 2);
    const socket = harness.sockets[1];
    sendStatus(socket);
    [0, 1, 2, 2].forEach(round => sendPoint(socket, round));
    ['starting', 'round two'].forEach(line => socket.receive(line));
    oldSocket.receive('old connection late log');
    sendPoint(oldSocket, 9);
    oldSocket.onclose?.();
    await harness.tick(200);
    await harness.tick(5000);
    assert.equal(harness.sockets.length, 2);
    harness.runFrames();
    assertCharts(harness, [0, 1, 2]);
    assert.deepEqual(harness.logs(), ['starting', 'round two']);
    assert.equal(harness.progress(), 20);
  } finally { harness.unmount(); }
});

test('实验已完成时刷新仍恢复完整历史及 100% 进度，不等待新数据', async () => {
  const harness = createHarness({ status: 'finished', exp_name: 'same_name', total_rounds: 10 });
  try {
    await harness.settle();
    const socket = harness.sockets[0];
    sendStatus(socket, 'finished');
    [0, 1, 10].forEach(round => sendPoint(socket, round));
    socket.receive('final summary');
    await harness.tick(200);
    harness.runFrames();
    assertCharts(harness, [0, 1, 10]);
    assert.equal(harness.progress(), 100);
    assert.deepEqual(harness.logs(), ['final summary']);
  } finally { harness.unmount(); }
});

test('后端切换活动实验时不混入前一个实验的图表与日志', async () => {
  const harness = createHarness();
  try {
    await harness.settle();
    const previous = harness.sockets[0];
    [0, 1, 2].forEach(round => sendPoint(previous, round));
    previous.receive('previous run');
    await harness.tick(200);
    harness.runFrames();
    harness.setStatus({ status: 'running', exp_name: 'another_name', total_rounds: 5 });
    await harness.tick(5000);
    assert.equal(previous.closed, true);
    const socket = harness.sockets[1];
    assert.equal(socket.url, 'ws://test/ws/logs/another_name');
    socket.receive({ type: 'status', data: { status: 'running', exp_name: 'another_name', total_rounds: 5 } });
    sendPoint(socket, 0);
    socket.receive('another run');
    await harness.tick(200);
    harness.runFrames();
    assertCharts(harness, [0]);
    assert.equal(harness.progress(), 0);
    assert.deepEqual(harness.logs(), ['another run']);
  } finally { harness.unmount(); }
});