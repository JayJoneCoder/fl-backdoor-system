import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compile = path => ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const componentCode = compile('../src/pages/BatchPage.tsx');
const requestCode = compile('../src/api/checkedFetch.ts');
const requests = {};
runInNewContext(requestCode, { exports: requests });

const createHarness = (initial = { status: 'idle' }, { delayed = false } = {}) => {
  const slots = [], effects = [], messages = [], starts = [];
  let cursor = 0, dirty = false, tree, resolveInitial;
  const initialResponse = delayed ? new Promise(resolve => { resolveInitial = resolve; }) : Promise.resolve({ data: initial });
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
  const nodes = () => {
    const found = [];
    const visit = node => {
      if (!node || typeof node !== 'object') return;
      if (Array.isArray(node)) { node.forEach(visit); return; }
      found.push(node);
      visit(node.props?.children);
    };
    visit(tree);
    return found;
  };
  const antd = Object.fromEntries(['Card', 'Upload', 'Button', 'Space', 'Table', 'Alert', 'Row', 'Col'].map(name => [name, name]));
  antd.Input = { TextArea: 'TextArea' };
  antd.message = Object.fromEntries(['success', 'error', 'info'].map(level => [level, text => messages.push({ level, text })]));
  const imports = {
    react, 'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' }, antd,
    '@ant-design/icons': { UploadOutlined: 'UploadIcon', PlayCircleOutlined: 'Play', EditOutlined: 'Edit', EyeOutlined: 'Eye' },
    '../api/client': {
      getBatchStatus: () => initialResponse,
      runBatchExperiments: async experiments => { starts.push(experiments); return { data: { batch_id: 'new' } }; },
      parseBatchConfig: async () => ({ data: { experiments: [{ name: 'new-one' }], count: 1 } }),
      cancelBatch: async () => ({ data: { status: 'cancelled' } }),
    },
    '../components/BatchMonitor': { __esModule: true, default: 'BatchMonitor' },
    '../api/checkedFetch': requests,
  };
  const exports = {};
  runInNewContext(componentCode, { exports, console, require: name => {
    assert.ok(name in imports, name); return imports[name];
  } });
  const render = () => {
    do {
      cursor = 0; dirty = false;
      tree = exports.default();
      effects.splice(0).forEach(callback => callback());
    } while (dirty);
  };
  const settle = async () => {
    for (let i = 0; i < 4; i++) { await Promise.resolve(); render(); }
  };
  const button = text => nodes().find(node => node.type === 'Button' && node.props.children === text);
  const monitor = () => nodes().find(node => node.type === 'BatchMonitor');
  render();
  return {
    settle, messages, starts, monitor,
    resolveInitial: value => resolveInitial({ data: value }),
    isRunning: () => !!button('取消整批／重试清理'),
    complete: (value, notify) => { monitor().props.onComplete(value, notify); render(); },
    start: async () => {
      const upload = nodes().find(node => node.type === 'Upload');
      upload.props.onChange({ fileList: [{ originFileObj: { text: async () => '{}' } }] });
      render();
      await button('解析文件').props.onClick(); render();
      await button('运行批量实验').props.onClick(); render();
    },
    unmount: () => slots.forEach(slot => slot?.cleanup?.()),
  };
};

const snapshot = (state, batch_id = 'old') => ({ status: state, batch_id, failed_count: 0,
  experiments: [{ name: 'one' }], error: 'Backend restarted; task was interrupted and was not automatically resumed.' });

test('进入或返回批量页面不恢复历史终态，不弹历史完成提示', async () => {
  for (const state of ['idle', 'completed', 'failed', 'cancelled']) {
    const h = createHarness(snapshot(state));
    try {
      await h.settle();
      assert.equal(h.monitor(), undefined);
      assert.equal(h.isRunning(), false);
      assert.deepEqual(h.messages, []);
    } finally { h.unmount(); }
  }
});

test('活动批次仍会恢复，实时结束通知一次且释放页面按钮', async () => {
  for (const state of ['pending', 'running', 'stopping']) {
    const h = createHarness(snapshot(state));
    try {
      await h.settle();
      assert.equal(h.monitor().props.batchId, 'old');
      assert.equal(h.isRunning(), true);
      h.complete({ ...snapshot('failed'), failed_count: 1 }, true);
      assert.equal(h.isRunning(), false);
      assert.equal(h.messages.length, 1);
      assert.match(h.messages[0].text, /批量实验有失败项/);
    } finally { h.unmount(); }
  }
});

test('恢复请求与任务结束竞争时只更新状态，不把历史终态当新失败', async () => {
  const h = createHarness(snapshot('running'));
  try {
    await h.settle();
    h.complete(snapshot('failed'), false);
    assert.equal(h.isRunning(), false);
    assert.deepEqual(h.messages, []);
  } finally { h.unmount(); }
});

test('本页面新启动的任务在首条状态前结束，也正常通知完成', async () => {
  for (const state of ['completed', 'failed', 'cancelled']) {
    const h = createHarness();
    try {
      await h.settle(); await h.start();
      assert.equal(h.monitor().props.batchId, 'new');
      assert.equal(h.isRunning(), true);
      h.messages.splice(0);
      h.complete(snapshot(state, 'new'), false);
      assert.equal(h.isRunning(), false);
      assert.equal(h.messages.length, 1);
      if (state === 'failed') {
        assert.match(h.messages[0].text, /批量任务未能完成/);
        assert.ok(!h.messages[0].text.includes('有失败项'));
      }
    } finally { h.unmount(); }
  }
});

test('迟到的页面恢复请求不能替换用户刚启动的新批次', async () => {
  const h = createHarness(undefined, { delayed: true });
  try {
    await h.start();
    h.resolveInitial(snapshot('running', 'stale'));
    await h.settle();
    assert.equal(h.monitor().props.batchId, 'new');
  } finally { h.unmount(); }
});

test('新批次已快速完成，迟到的恢复请求也不能重新挂上旧批次', async () => {
  const h = createHarness(undefined, { delayed: true });
  try {
    await h.start();
    h.complete(snapshot('completed', 'new'), false);
    h.resolveInitial(snapshot('running', 'stale'));
    await h.settle();
    assert.equal(h.monitor().props.batchId, 'new');
    assert.equal(h.isRunning(), false);
  } finally { h.unmount(); }
});

test('卸载页面后的恢复响应不更新已卸载页面', async () => {
  const h = createHarness(undefined, { delayed: true });
  h.unmount();
  h.resolveInitial(snapshot('running'));
  await h.settle();
  assert.equal(h.monitor(), undefined);
});