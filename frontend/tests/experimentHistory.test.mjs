import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compile = path => ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const componentCode = compile('../src/components/ExperimentHistory/index.tsx');
const requestCode = compile('../src/api/checkedFetch.ts');
const uuid = 'f7f8c75e-32a8-485b-ab2b-e0db3f1677ea';
const summary = (name, runId) => ({ name, run_id: runId, accuracy: 0.95, asr: 0.02, loss: 0.1, last_round: 3 });
const detail = (name, runId) => ({ name, run_id: runId, metrics: summary(name), images: [], csv_files: [] });

const createHarness = (experiments, details = {}) => {
  const slots = [], effects = [], errors = [], requests = [];
  let cursor = 0, dirty = false, tree;
  const react = {
    useState: initial => {
      const index = cursor++;
      const slot = slots[index] ??= { value: initial };
      return [slot.value, value => { slot.value = value; dirty = true; }];
    },
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
  const findAll = (root, type) => {
    const nodes = [];
    visit(root, node => { if (node.type === type) nodes.push(node); });
    return nodes;
  };
  const text = node => {
    if (typeof node === 'string' || typeof node === 'number') return String(node);
    if (!node || typeof node !== 'object') return '';
    if (Array.isArray(node)) return node.map(text).join('');
    return text(node.props?.children);
  };
  const load = (source, imports) => {
    const exports = {};
    runInNewContext(source, { exports, require: name => {
      assert.ok(name in imports, `Unexpected import: ${name}`);
      return imports[name];
    } });
    return exports;
  };
  const antd = Object.fromEntries(['Card', 'Row', 'Col', 'Spin', 'Button', 'Modal', 'Table',
    'Space', 'Divider', 'Image', 'Statistic', 'Descriptions', 'Tooltip', 'Dropdown'].map(name => [name, name]));
  antd.Typography = { Text: 'Typography.Text' };
  antd.message = { error: value => errors.push(value), success() {} };
  const icons = Object.fromEntries(['BarChartOutlined', 'DownloadOutlined', 'FileTextOutlined',
    'ExperimentOutlined', 'AimOutlined', 'FileImageOutlined', 'FileZipOutlined'].map(name => [name, name]));
  const charts = Object.fromEntries(['LineChart', 'Line', 'XAxis', 'YAxis', 'CartesianGrid',
    'Tooltip', 'Legend', 'ResponsiveContainer'].map(name => [name, `Recharts.${name}`]));
  const failOrReturn = value => {
    if (value instanceof Error || value?.response) throw value;
    return { data: value };
  };
  const api = {
    listExperiments: async () => failOrReturn(experiments),
    getExperimentDetail: async name => { requests.push(name); return failOrReturn(details[name]); },
    generatePlots: async () => {}, downloadExperimentImages: async () => {}, downloadExperimentAllFiles: async () => {},
  };
  const component = load(componentCode, {
    react, 'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' }, antd,
    '@ant-design/icons': icons, recharts: charts, '../../api/client': api,
    '../../api/endpoints': { getBackendUrl: value => value },
    '../../api/checkedFetch': load(requestCode, {}),
  }).default;
  const render = () => {
    do {
      cursor = 0; dirty = false;
      tree = component();
      effects.splice(0).forEach(callback => callback());
    } while (dirty);
  };
  const settle = async () => {
    for (let i = 0; i < 4; i++) { await Promise.resolve(); render(); }
  };
  render();
  return { settle, errors, requests, text,
    cards: () => findAll(tree, 'Card').filter(node => node.props.hoverable),
    modal: () => findAll(tree, 'Modal')[0],
    texts: root => findAll(root, 'Typography.Text'),
    codes: root => findAll(root, 'code'),
    tooltips: root => findAll(root, 'Tooltip'),
  };
};

test('历史卡片显示实验 UUID，省略长文本且悬停可查看完整值', async () => {
  const h = createHarness([summary('新实验', uuid)]);
  await h.settle();
  const card = h.cards()[0];
  const label = h.texts(card).find(node => h.text(node).startsWith('UUID：'));
  assert.equal(h.text(label), `UUID：${uuid}`);
  assert.equal(label.props.ellipsis, true);
  assert.equal(label.props.type, 'secondary');
  assert.ok(h.tooltips(card).some(node => node.props.title === uuid));
});

test('打开详情后以黑色和浅色背景显示完整 UUID，复制使用完整原值和中文提示', async () => {
  const h = createHarness([summary('新实验', uuid)], { 新实验: detail('新实验', uuid) });
  await h.settle();
  await h.cards()[0].props.onClick();
  await h.settle();
  assert.equal(h.modal().props.open, true);
  const label = h.texts(h.modal()).find(node => node.props.copyable);
  assert.equal(h.text(label), uuid);
  assert.equal(label.props.copyable.text, uuid);
  assert.deepEqual([...label.props.copyable.tooltips], ['复制 UUID', '已复制']);
  assert.equal(label.props.style.wordBreak, 'break-all');
  assert.equal(label.props.style.color, '#000');
  const code = h.codes(label)[0];
  assert.ok(code, 'UUID 使用显式的 code 节点，避免全局深色样式覆盖文字颜色');
  assert.equal(h.text(code), uuid);
  assert.equal(code.props.style.color, '#000');
  assert.equal(code.props.style.background, '#f5f5f5');
  assert.deepEqual(h.requests, ['新实验']);
});

test('旧实验没有 UUID 时显示说明，不伪造 UUID 或提供无效复制按钮', async () => {
  for (const missing of [undefined, null, '']) {
    const h = createHarness([summary('旧实验', missing)], { 旧实验: detail('旧实验', missing) });
    await h.settle();
    assert.match(h.text(h.cards()[0]), /UUID：未记录/);
    await h.cards()[0].props.onClick();
    await h.settle();
    assert.match(h.text(h.modal()), /未记录 UUID（旧实验）/);
    const label = h.texts(h.modal()).find(node => h.text(node) === '未记录 UUID（旧实验）');
    assert.equal(label.props.style.color, '#000');
    assert.equal(h.codes(label).length, 0);
    assert.ok(!h.texts(h.modal()).some(node => node.props.copyable));
  }
});

test('历史列表和详情接口失败时给出中文错误，不把底层英文直接弹给用户', async () => {
  const list = createHarness({ response: { status: 500, data: { detail: 'private traceback' } } });
  await list.settle();
  assert.match(list.errors[0], /后端处理失败（HTTP 500）/);
  const h = createHarness([summary('新实验', uuid)], { 新实验: new Error('private traceback') });
  await h.settle();
  await h.cards()[0].props.onClick();
  await h.settle();
  assert.equal(h.errors[0], '加载实验详情失败，请稍后重试。');
  assert.ok(!h.text(h.modal()).includes(uuid));
});