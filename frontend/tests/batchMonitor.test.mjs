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

const createHarness = () => {
  const slots = [], effects = [], sockets = [], notifications = [], notificationFlags = [];
  let cursor = 0, dirty = false, tree, lines = [];
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
    status(status) { this.onmessage({ data: JSON.stringify({ type: 'status', data: status }) }); }
  }
  const exports = {};
  const imports = {
    react,
    'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' },
    antd: { Card: 'Card', Progress: 'Progress', Tag: 'Tag', Alert: 'Alert', Space: 'Space', Typography: { Text: 'Text' } },
    '@ant-design/icons': { SyncOutlined: 'Sync', CheckCircleOutlined: 'Check', CloseCircleOutlined: 'Close' },
    '../LogViewer': { __esModule: true, default: 'LogViewer' },
    '../../api/endpoints': { getWebSocketUrl: value => value },
    '../../hooks/useBufferedLogs': { useBufferedLogs: () => ({ logs: lines, queueLog, flushLogs, resetLogs }) },
    '../../api/checkedFetch': requests,
  };
  runInNewContext(source, { exports, WebSocket: Socket, require: name => {
    assert.ok(name in imports, name); return imports[name];
  } });
  const onComplete = (status, notify) => {
    notifications.push(JSON.parse(JSON.stringify(status)));
    notificationFlags.push(notify);
  };
  const render = () => {
    do {
      cursor = 0; dirty = false;
      tree = exports.default({ batchId: 'batch', onComplete });
      effects.splice(0).forEach(callback => callback());
    } while (dirty);
  };
  render();
  return { sockets, notifications, notificationFlags, status: status => { sockets[0].status(status); render(); },
    progress: () => { let node; visit(tree, item => { if (item.type === 'Progress') node = item; }); return node.props; },
    alert: () => { let node; visit(tree, item => { if (item.type === 'Alert') node = item; }); return node?.props.message; },
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

test('批量失败原因和逐项状态以中文显示，完成回调保留原始诊断', () => {
  const h = createHarness();
  try {
    const error = 'Experiment worker exited with code 1. See run.log for details.';
    h.status({ ...status('failed', 2), error, results: [{ name: 'one', status: 'failed', exit_code: 1, error }] });
    assert.match(h.alert(), /实验运行失败（退出码 1）/);
    assert.match(h.resultLines(), /one：失败/);
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
    assert.match(h.resultLines(), /old：未记录/);
    assert.match(h.resultLines(), /无法确认该实验的执行状态/);
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