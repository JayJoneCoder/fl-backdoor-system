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
const batchUtils = {};
runInNewContext(compile('../src/utils/batchStream.ts'), { exports: batchUtils, require: () => requests });

const createHarness = (initial = { status: 'idle' }, { delayed = false, initialError = null, controlDeferred = false } = {}) => {
  const slots = [], effects = [], messages = [], starts = [], controls = [];
  let cursor = 0, dirty = false, tree, resolveInitial, resolveControl, parseError = null;
  const controlResponse = controlDeferred ? new Promise(resolve => { resolveControl = resolve; }) : null;
  const parses = [], scrolls = [];
  const initialResponse = delayed ? new Promise(resolve => { resolveInitial = resolve; }) : initialError ? Promise.reject(initialError) : Promise.resolve({ data: initial });
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
      visit(node.props?.extra);
      visit(node.props?.children);
    };
    visit(tree);
    return found;
  };
  const antd = Object.fromEntries(['Card', 'Upload', 'Button', 'Space', 'Table', 'Alert', 'Row', 'Col', 'Tag', 'Popconfirm', 'Select'].map(name => [name, name]));
  antd.Typography = { Text: 'Text' };
  antd.Input = { TextArea: 'TextArea' };
  antd.message = Object.fromEntries(['success', 'error', 'info'].map(level => [level, text => messages.push({ level, text })]));
  const imports = {
    react, 'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' }, antd,
    '@ant-design/icons': { UploadOutlined: 'UploadIcon', PlayCircleOutlined: 'Play', EditOutlined: 'Edit', EyeOutlined: 'Eye' },
    '../api/client': {
      getBatchStatus: () => initialResponse,
      runBatchExperiments: async experiments => { starts.push(experiments); return { data: { batch_id: 'new' } }; },
      parseBatchConfig: async file => {
        parses.push(await file.text());
        if (parseError) throw parseError;
        return { data: { experiments: [{ name: 'new-one' }], count: 1 } };
      },
      cancelBatch: async () => ({ data: { status: 'cancelled' } }),
      pauseBatch: async id => { controls.push(['pause', id]); return controlResponse ?? { data: { ...initial, pause_requested: true } }; },
      resumeBatch: async id => { controls.push(['resume', id]); return { data: { ...initial, pause_requested: false } }; },
      stopCurrentBatchExperiment: async (id, runId) => { controls.push(['stop-current', id, runId]); return { data: initial }; },
    },
    '../components/BatchMonitor': { __esModule: true, default: 'BatchMonitor' },
    '../api/checkedFetch': requests,
    '../utils/batchStream': batchUtils,
  };
  const exports = {};
  runInNewContext(componentCode, { exports, console, Blob, File,
    requestAnimationFrame: callback => callback(), require: name => {
    assert.ok(name in imports, name); return imports[name];
  } });
  const render = () => {
    do {
      cursor = 0; dirty = false;
      tree = exports.default();
      const table = nodes().find(node => node.type === 'Table');
      if (table?.props.ref) table.props.ref.current = { scrollTo: value => scrolls.push(value) };
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
    settle, messages, starts, monitor, nodes, controls, parses, scrolls, button,
    resolveControl: value => resolveControl({ data: value }),
    failParse: error => { parseError = error; },
    edit: value => { nodes().find(node => node.type === 'TextArea').props.onChange({ target: { value } }); render(); },
    filter: value => { nodes().find(node => node.type === 'Select').props.onChange(value); render(); },
    resolveInitial: value => resolveInitial({ data: value }),
    isRunning: () => !!button('结束整批／重试清理'),
    complete: (value, notify) => { monitor().props.onComplete(value, notify); render(); },
    update: value => { monitor().props.onStatus(value); render(); },
    progress: value => { monitor().props.onProgress(value); render(); },
    click: async text => { await button(text).props.onClick(); render(); },
    openConfirm: title => { nodes().find(node => node.type === 'Popconfirm' && node.props.title === title).props.onOpenChange?.(true); render(); },
    confirm: async (title, open = true) => {
      const popup = nodes().find(node => node.type === 'Popconfirm' && node.props.title === title);
      if (open) popup.props.onOpenChange?.(true);
      await popup.props.onConfirm(); render();
    },
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
  for (const state of ['pending', 'running', 'paused', 'stopping']) {
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

test('恢复批次无需上传文件即可显示带序号和状态的可折叠列表，且禁止替换运行配置', async () => {
  const h = createHarness(snapshot('running'));
  try {
    await h.settle();
    const table = h.nodes().find(node => node.type === 'Table');
    assert.ok(table);
    assert.equal(table.props.columns[0].title, '序号');
    assert.equal(table.props.columns[0].render(null, {}, 0), 1);
    assert.equal(table.props.columns[0].render(null, {}, 12), 13);
    assert.equal(table.props.columns.at(-1).title, '实验状态');
    assert.equal(h.nodes().find(node => node.type === 'Upload').props.disabled, true);
    const card = () => h.nodes().find(node => node.type === 'Card' && node.props.title === '实验列表 (1)');
    assert.equal(card().props.extra.props['aria-expanded'], true);
    await h.click('收起');
    assert.equal(card().props.styles.body.display, 'none');
    await h.click('展开');
    assert.equal(card().props.extra.props['aria-expanded'], true);
  } finally { h.unmount(); }
});

test('实时指标按运行标识同步到列表，失败显示退出码与简明原因', async () => {
  const h = createHarness(snapshot('running'));
  try {
    await h.settle();
    h.update({ ...snapshot('running'), results: [{ name: 'one', status: 'running', run_id: 'run1' }] });
    h.progress({ runId: 'run1', percent: 40 });
    const display = () => h.nodes().find(node => node.type === 'Table').props.columns.at(-1).render(null, {}, 0);
    assert.equal(display().props.children[0].props.children, '正在进行，进度 40%');
    h.progress({ runId: 'stale', percent: 90 });
    assert.equal(display().props.children[0].props.children, '正在进行，进度 0%');
    h.update({ ...snapshot('failed'), results: [{ name: 'one', status: 'failed', exit_code: 3, error: 'Experiment worker exited with code 3. See run.log for details.' }] });
    assert.equal(display().props.children[0].props.children, '失败');
    assert.match(display().props.children[1].props.children, /退出码 3/);
  } finally { h.unmount(); }
});

test('暂停与继续队列调用独立接口，仅结束当前实验必须带上当前运行标识', async () => {
  const initial = { ...snapshot('running'), current_index: 0, current_exp_name: 'one',
    results: [{ name: 'one', status: 'running', run_id: 'run1' }] };
  const h = createHarness(initial);
  try {
    await h.settle();
    await h.click('暂停队列');
    await h.click('继续队列');
    await h.confirm('仅结束当前实验？');
    assert.deepEqual(h.controls, [['pause', 'old'], ['resume', 'old'], ['stop-current', 'old', 'run1']]);
  } finally { h.unmount(); }
});

test('确认框打开后实验切换，停止请求仍绑定原实验，不误停止下一项', async () => {
  const initial = { ...snapshot('running'), current_index: 0, current_exp_name: 'one',
    results: [{ name: 'one', status: 'running', run_id: 'run1' }] };
  const h = createHarness(initial);
  try {
    await h.settle();
    h.openConfirm('仅结束当前实验？');
    h.update({ ...initial, current_index: 1, current_exp_name: 'two', results: [
      { ...initial.results[0], status: 'completed' }, { name: 'two', status: 'running', run_id: 'run2' },
    ] });
    await h.confirm('仅结束当前实验？', false);
    assert.deepEqual(h.controls, [['stop-current', 'old', 'run1']]);
  } finally { h.unmount(); }
});

test('迟到的暂停响应不能把已完成批次重新变为运行中', async () => {
  const h = createHarness({ ...snapshot('running'), revision: 1 }, { controlDeferred: true });
  try {
    await h.settle();
    const request = h.click('暂停队列');
    h.complete({ ...snapshot('completed'), revision: 3 }, true);
    h.resolveControl({ ...snapshot('running'), revision: 2, pause_requested: true });
    await request;
    assert.equal(h.isRunning(), false);
    assert.equal(h.messages.length, 1);
  } finally { h.unmount(); }
});

test('同一实验内的旧暂停版本不能覆盖较新的继续队列状态', async () => {
  const h = createHarness({ ...snapshot('running'), revision: 1 }, { controlDeferred: true });
  try {
    await h.settle();
    const request = h.click('暂停队列');
    h.update({ ...snapshot('running'), revision: 3, pause_requested: false });
    h.resolveControl({ ...snapshot('running'), revision: 2, pause_requested: true });
    await request;
    assert.ok(h.button('暂停队列'));
    h.update({ ...snapshot('running'), revision: 2, pause_requested: true });
    assert.ok(h.button('暂停队列'));
  } finally { h.unmount(); }
});

test('卸载后的操作响应和旧批次消息不能更新状态或提示', async () => {
  const h = createHarness(snapshot('running'), { controlDeferred: true });
  await h.settle();
  h.update(snapshot('failed', 'another-batch'));
  assert.equal(h.isRunning(), true);
  const request = h.click('暂停队列');
  h.unmount();
  h.resolveControl({ ...snapshot('running'), pause_requested: true });
  await request;
  assert.deepEqual(h.messages, []);
});

test('恢复配置可编辑，修改预览先解析，解析失败保留编辑且不能运行，取消还原', async () => {
  const h = createHarness(snapshot('running'));
  try {
    await h.settle(); h.complete(snapshot('completed'), false);
    assert.equal(h.button('编辑 JSON').props.disabled, false);
    await h.click('编辑 JSON');
    const initialText = h.nodes().find(node => node.type === 'TextArea').props.value;
    assert.deepEqual(JSON.parse(initialText).experiments, [{ name: 'one' }]);
    h.edit('[null]');
    assert.equal(h.button('运行批量实验').props.disabled, true);
    h.failParse({ response: { data: { detail: '第 1 项实验必须为 JSON 对象。' } } });
    await h.click('预览表格');
    assert.equal(h.parses.at(-1), '[null]');
    assert.ok(h.nodes().find(node => node.type === 'TextArea'));
    await h.click('运行批量实验');
    assert.equal(h.starts.length, 0);
    await h.click('取消编辑'); await h.click('编辑 JSON');
    assert.equal(h.nodes().find(node => node.type === 'TextArea').props.value, initialText);
    h.failParse(null); h.edit('[{"name":"changed"}]');
    await h.click('预览表格');
    assert.equal(h.nodes().find(node => node.type === 'TextArea'), undefined);
    assert.equal(h.nodes().find(node => node.type === 'Table').props.dataSource[0].name, 'new-one');
    assert.equal(h.button('运行批量实验').props.disabled, false);
  } finally { h.unmount(); }
});

test('筛选状态保留原始序号与错误，定位当前实验恢复全部项并滚动', async () => {
  const h = createHarness({ ...snapshot('running'), current_index: 2, current_exp_name: 'three',
    experiments: [{ name: 'one' }, { name: 'two' }, { name: 'three' }], results: [
      { name: 'one', status: 'completed' }, { name: 'two', status: 'failed', exit_code: 7 }, { name: 'three', status: 'running' },
    ] });
  try {
    await h.settle(); h.filter('failed');
    const table = h.nodes().find(node => node.type === 'Table');
    assert.equal(table.props.dataSource.length, 1);
    const record = table.props.dataSource[0];
    assert.equal(table.props.columns[0].render(null, record, 0), 2);
    const display = table.props.columns.at(-1).render(null, record, 0);
    assert.equal(display.props.children[0].props.children, '失败');
    await h.click('定位当前实验');
    assert.equal(h.nodes().find(node => node.type === 'Select').props.value, 'all');
    assert.equal(h.scrolls.at(-1).key, 'three');
  } finally { h.unmount(); }
});

test('恢复请求失败明确提示且可手动重试，不冒充空闲状态', async () => {
  const h = createHarness(undefined, { initialError: { code: 'ERR_NETWORK' } });
  try {
    await h.settle();
    const warning = h.nodes().find(node => node.type === 'Alert' && node.props.type === 'warning');
    assert.match(warning.props.title, /无法确认活动批次/);
    assert.equal(warning.props.action.props.children, '重试检查');
  } finally { h.unmount(); }
});