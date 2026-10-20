import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const source = ts.transpileModule(readFileSync(new URL('../src/components/BatchMonitor/index.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const requestSource = ts.transpileModule(readFileSync(new URL('../src/api/checkedFetch.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023 },
}).outputText;
const requests = {};
runInNewContext(requestSource, { exports: requests });
const utils = {};
runInNewContext(ts.transpileModule(readFileSync(new URL('../src/utils/batchStream.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023 },
}).outputText, { exports: utils, require: () => requests });

const createHarness = () => {
  const slots = [], effects = [], sockets = [], notifications = [], notificationFlags = [], updates = [];
  const timers = new Map();
  let cursor = 0, dirty = false, tree, lines = [];
  let accept = true;
  const react = {
    useState: initial => {
      const index = cursor++;
      const slot = slots[index] ??= { value: initial };
      return [slot.value, value => { slot.value = value; dirty = true; }];
    },
    useRef: initial => { const index = cursor++; return slots[index] ??= { current: initial }; },
    useEffect: (callback, deps) => {
      const index = cursor++;
      const previous = slots[index];
      if (!previous || deps.some((value, i) => value !== previous.deps[i])) {
        const slot = slots[index] = { deps };
        effects.push(() => { previous?.cleanup?.(); slot.cleanup = callback(); });
      }
    },
  };
  const jsx = (type, props) => ({ type, props });
  const visit = (node, callback) => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { node.forEach(item => visit(item, callback)); return; }
    callback(node);
    visit(node.props?.children, callback);
  };
  const queueLog = line => lines.push(line);
  const flushLogs = () => {};
  const resetLogs = () => { lines = []; };
  class Socket {
    constructor() { sockets.push(this); }
    close() { this.closed = true; }
    disconnect() { this.onclose?.(); }
    status(status) { this.onmessage({ data: JSON.stringify({ type: 'status', data: status }) }); }
  }
  const exports = {};
  const imports = {
    react,
    'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' },
    antd: { Card: 'Card', Progress: 'Progress', Tag: 'Tag', Alert: 'Alert', Space: 'Space', Row: 'Row', Col: 'Col', Button: 'Button', Typography: { Text: 'Text' } },
    '@ant-design/icons': { SyncOutlined: 'Sync', CheckCircleOutlined: 'Check', CloseCircleOutlined: 'Close' },
    '../LogViewer': { __esModule: true, default: 'LogViewer' },
    '../../api/endpoints': { getWebSocketUrl: value => value },
    '../../hooks/useBufferedLogs': { useBufferedLogs: () => ({ logs: lines, queueLog, flushLogs, resetLogs }) },
    '../../api/checkedFetch': requests,
    '../../utils/batchStream': utils,
    './ExperimentMetrics': { __esModule: true, default: 'ExperimentMetrics' },
  };
  runInNewContext(source, { exports, WebSocket: Socket,
    setTimeout: callback => { const id = Symbol(); timers.set(id, callback); return id; },
    clearTimeout: id => timers.delete(id), require: name => {
    assert.ok(name in imports, name); return imports[name];
  } });
  const onComplete = (status, notify) => {
    notifications.push(JSON.parse(JSON.stringify(status)));
    notificationFlags.push(notify);
  };
  const onStatus = value => { if (!accept) return false; updates.push(value); };
  const render = () => {
    do {
      cursor = 0; dirty = false;
      tree = exports.default({ batchId: 'batch', onComplete, onStatus });
      effects.splice(0).forEach(callback => callback());
    } while (dirty);
  };
  render();
  const all = type => { const found = []; visit(tree, item => { if (item.type === type) found.push(item); }); return found; };
  return { sockets, notifications, notificationFlags, updates, render,
    accept: value => { accept = value; }, status: status => { sockets.at(-1).status(status); render(); },
    progress: () => all('Progress')[0].props,
    currentProgress: () => all('Progress')[1].props,
    metric: () => all('ExperimentMetrics')[0],
    metricUpdate: value => { all('ExperimentMetrics')[0].props.onProgress(value); render(); },
    reconnect: () => { [...timers.values()].forEach(callback => callback()); timers.clear(); render(); },
    alert: () => { let node; visit(tree, item => { if (item.type === 'Alert') node = item; }); return node?.props.title; },
    resultLines: () => { const values = []; visit(tree, item => { if (item.type === 'Text') values.push(item.props.children); }); return values.flat().filter(value => typeof value === 'string').join(''); },
    unmount: () => slots.forEach(slot => slot?.cleanup?.()) };
};

const status = (state, completed = 0) => ({ batch_id: 'batch', status: state, total: 2, current_index: 1,
  current_exp_name: 'one', error: null, completed_count: completed, success_count: 0, failed_count: 0, cancelled_count: 0, results: [] });

test('批量进度按已结束项计算，不使用从零开始的当前索引', () => {
  const h = createHarness();
  try {
    h.status(status('running', 0)); assert.equal(h.progress().percent, 0);
    h.status(status('running', 1)); assert.equal(h.progress().percent, 50);
  } finally { h.unmount(); }
});

test('批量失败原因以中文显示，逐项状态仅交给列表，完成回调保留原始诊断', () => {
  const h = createHarness();
  try {
    const error = 'Experiment worker exited with code 1. See run.log for details.';
    h.status({ ...status('failed', 2), error, results: [{ name: 'one', status: 'failed', exit_code: 1, error }] });
    assert.match(h.alert(), /实验运行失败（退出码 1）/);
    assert.ok(!h.resultLines().includes('one：失败'));
    assert.ok(!h.resultLines().includes('Experiment worker'));
    assert.equal(h.notifications[0].error, error);
  } finally { h.unmount(); }
});

test('停止／清理状态不能提前通知完成，取消和失败携带真实状态只通知一次', () => {
  for (const terminal of ['failed', 'cancelled', 'completed']) {
    const h = createHarness();
    try {
      h.status(status('stopping')); assert.equal(h.notifications.length, 0);
      h.status(status(terminal, 2)); h.status(status(terminal, 2));
      assert.equal(h.notifications.length, 1);
      assert.equal(h.notifications[0].status, terminal);
      assert.deepEqual(h.notificationFlags, [true]);
      assert.equal(h.progress().status, terminal === 'failed' ? 'exception' : terminal === 'completed' ? 'success' : 'normal');
    } finally { h.unmount(); }
  }
});

test('首次收到历史终态只同步父页面状态，不请求弹出完成通知', () => {
  for (const terminal of ['completed', 'failed', 'cancelled']) {
    const h = createHarness();
    try {
      h.status(status(terminal, 2));
      h.status(status(terminal, 2));
      assert.equal(h.notifications.length, 1, '父页面必须解除忙碌状态');
      assert.deepEqual(h.notificationFlags, [false]);
    } finally { h.unmount(); }
  }
});

test('旧记录未记录的逐项状态不用取消冒充，界面提示为中文', () => {
  const h = createHarness();
  try {
    h.status({ ...status('failed'), unknown_count: 2, results: [
      { name: 'old', status: 'unknown', exit_code: null, error: 'Execution details were not recorded by this older version.' },
    ] });
    assert.match(h.resultLines(), /状态未记录/);
    assert.ok(!h.resultLines().includes('old：未记录'), '不再堆叠等待队列或逐项状态');
    assert.ok(!h.resultLines().includes('未启动'));
    assert.equal(h.progress().percent, 0);
  } finally { h.unmount(); }
});

test('卸载后的旧批量消息不能再次触发完成通知', () => {
  const h = createHarness();
  h.unmount();
  h.sockets[0].status(status('completed', 2));
  assert.equal(h.notifications.length, 0);
  assert.equal(h.sockets[0].closed, true);
});

test('双进度独立更新，指标连接按运行标识切换并在结束后保留最后一次', () => {
  const h = createHarness();
  try {
    const first = { name: 'one', status: 'running', run_id: 'r1' };
    h.status({ ...status('running'), current_index: 0, results: [first] });
    assert.equal(h.metric().props.runId, 'r1');
    h.metricUpdate({ runId: 'r1', round: 4, total: 10, percent: 40 });
    assert.equal(h.progress().percent, 0);
    assert.equal(h.currentProgress().percent, 40);
    const results = [{ ...first, status: 'completed' }, { name: 'two', status: 'running', run_id: 'r2' }];
    h.status({ ...status('running', 1), current_exp_name: 'two', results });
    assert.equal(h.progress().percent, 50);
    assert.equal(h.currentProgress().percent, 0);
    assert.equal(h.metric().props.runId, 'r2');
    h.status({ ...status('completed', 2), current_exp_name: null, current_index: 2, results: results.map(item => ({ ...item, status: 'completed' })) });
    assert.equal(h.metric().props.runId, 'r2');
    assert.equal(h.currentProgress().percent, 100);
    assert.equal(h.updates.length, 3);
  } finally { h.unmount(); }
});

test('暂停是活动状态，断线自动重连，结束后不再重连且完成只通知一次', () => {
  const h = createHarness();
  try {
    h.status({ ...status('paused'), pause_requested: true });
    assert.equal(h.notifications.length, 0);
    h.sockets[0].disconnect(); h.reconnect();
    assert.equal(h.sockets.length, 2);
    h.status(status('completed', 2));
    assert.deepEqual(h.notificationFlags, [true]);
    h.sockets[1].disconnect(); h.reconnect();
    assert.equal(h.sockets.length, 2);
  } finally { h.unmount(); }
});

test('重连后旧连接消息、错误和关闭不能干扰新连接', () => {
  const h = createHarness();
  try {
    h.status({ ...status('running'), revision: 1 });
    h.sockets[0].disconnect(); h.reconnect();
    h.status({ ...status('paused'), revision: 3 });
    h.sockets[0].status({ ...status('failed', 2), revision: 4 });
    h.sockets[0].onerror(); h.sockets[0].disconnect(); h.render(); h.reconnect();
    assert.equal(h.sockets.length, 2);
    assert.equal(h.notifications.length, 0);
    assert.equal(h.updates.at(-1).status, 'paused');
    assert.equal(h.alert(), undefined);
  } finally { h.unmount(); }
});

test('版本回退、其他批次及终态之后的活动状态均被忽略', () => {
  const h = createHarness();
  try {
    h.status({ ...status('running'), revision: 4 });
    h.status({ ...status('paused'), revision: 3 });
    h.status({ ...status('failed'), batch_id: 'other', revision: 5 });
    assert.equal(h.updates.length, 1);
    h.status({ ...status('completed', 2), revision: 6 });
    h.status({ ...status('running'), revision: 7 });
    assert.equal(h.updates.at(-1).status, 'completed');
    assert.equal(h.notifications.length, 1);
  } finally { h.unmount(); }
});

test('父页面已有较新控制响应时，监控不得展示被父页面拒绝的旧推送', () => {
  const h = createHarness();
  try {
    h.status({ ...status('running'), revision: 1 });
    h.accept(false);
    h.status({ ...status('failed', 2), revision: 2 });
    assert.equal(h.notifications.length, 0);
    assert.equal(h.progress().percent, 0);
    h.accept(true);
    h.status({ ...status('completed', 2), revision: 3 });
    assert.equal(h.progress().percent, 100);
  } finally { h.unmount(); }
});