import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const source = ts.transpileModule(readFileSync(new URL('../src/components/BatchMonitor/index.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

const createHarness = () => {
  const slots = [], effects = [], sockets = [], notifications = [];
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
  };
  runInNewContext(source, { exports, WebSocket: Socket, require: name => {
    assert.ok(name in imports, name); return imports[name];
  } });
  const onComplete = status => notifications.push(JSON.parse(JSON.stringify(status)));
  const render = () => {
    do {
      cursor = 0; dirty = false;
      tree = exports.default({ batchId: 'batch', onComplete });
      effects.splice(0).forEach(callback => callback());
    } while (dirty);
  };
  render();
  return { sockets, notifications, status: status => { sockets[0].status(status); render(); },
    progress: () => { let node; visit(tree, item => { if (item.type === 'Progress') node = item; }); return node.props; },
    unmount: () => slots.forEach(slot => slot?.cleanup?.()) };
};

const status = (state, completed = 0) => ({ batch_id: 'batch', status: state, total: 2, current_index: 1,
  current_exp_name: 'one', error: null, completed_count: completed, success_count: 0, failed_count: 0, cancelled_count: 0, results: [] });

test('批量进度按已处理项计算，不使用从零开始的当前索引', () => {
  const h = createHarness();
  try {
    h.status(status('running', 0)); assert.equal(h.progress().percent, 0);
    h.status(status('running', 1)); assert.equal(h.progress().percent, 50);
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
      assert.equal(h.progress().status, terminal === 'failed' ? 'exception' : terminal === 'completed' ? 'success' : 'normal');
    } finally { h.unmount(); }
  }
});

test('卸载后的旧批量消息不能再次触发完成通知', () => {
  const h = createHarness();
  h.unmount();
  h.sockets[0].status(status('completed', 2));
  assert.equal(h.notifications.length, 0);
  assert.equal(h.sockets[0].closed, true);
});