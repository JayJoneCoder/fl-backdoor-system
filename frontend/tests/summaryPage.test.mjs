import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compile = path => ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const source = compile('../src/pages/SummaryPage.tsx');
const errorsSource = compile('../src/api/checkedFetch.ts');

function harness(download, generate = async () => {}) {
  const slots = [], effects = [], errors = [], confirms = [], saved = [];
  let cursor = 0, dirty = false, tree;
  const jsx = (type, props) => ({ type, props });
  const react = {
    useState(initial) {
      const index = cursor++;
      const slot = slots[index] ??= { value: initial };
      return [slot.value, value => { slot.value = value; dirty = true; }];
    },
    useEffect(callback) { if (!slots[cursor++]) { slots[cursor - 1] = true; effects.push(callback); } },
  };
  const antd = Object.fromEntries(['Card', 'Button', 'Table', 'Modal', 'Checkbox', 'Space', 'Input',
    'Image', 'Row', 'Col', 'Spin', 'Divider', 'Dropdown'].map(name => [name, name]));
  antd.Modal = Object.assign(() => {}, { confirm: value => confirms.push(value) });
  antd.message = { error: value => errors.push(value), success() {}, warning() {} };
  const icons = Object.fromEntries(['FileTextOutlined', 'PlusOutlined', 'BarChartOutlined', 'CodeOutlined',
    'DownloadOutlined', 'FileZipOutlined', 'FileImageOutlined'].map(name => [name, name]));
  const charts = Object.fromEntries(['LineChart', 'Line', 'XAxis', 'YAxis', 'CartesianGrid', 'Tooltip',
    'Legend', 'ResponsiveContainer'].map(name => [name, name]));
  const load = (code, imports) => {
    const exports = {};
    runInNewContext(code, { exports, Blob,
      window: { URL: { createObjectURL: () => 'blob:zip', revokeObjectURL() {} } },
      document: { body: { appendChild() {}, removeChild() {} },
        createElement: () => ({ click() { saved.push(this.download); } }) },
      require: name => { assert.ok(name in imports, name); return imports[name]; } });
    return exports;
  };
  const component = load(source, {
    react, 'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' }, antd,
    '@ant-design/icons': icons, recharts: charts,
    '../api/endpoints': { getBackendUrl: path => path },
    '../api/checkedFetch': load(errorsSource, {}),
    '../api/client': {
      listExperiments: async () => ({ data: [{ name: 'healthy', has_metrics: true },
        { name: 'failed', has_metrics: false }, { name: 'legacy' }] }),
      listSummaries: async () => ({ data: [{ name: 'report', experiments: ['healthy'] }] }),
      createSummary: async () => {}, generateSummaryPlots: generate, downloadSummaryZip: download,
      getSummaryDetail: async () => ({ data: {} }),
    },
  }).default;
  const render = () => {
    do { cursor = 0; dirty = false; tree = component(); effects.splice(0).forEach(fn => fn()); } while (dirty);
  };
  const find = type => {
    const nodes = [];
    const walk = value => {
      if (!value || typeof value !== 'object') return;
      if (Array.isArray(value)) return value.forEach(walk);
      if (value.type === type) nodes.push(value);
      walk(value.props?.children);
    };
    walk(tree); return nodes;
  };
  const settle = async () => { for (let i = 0; i < 6; i++) { await Promise.resolve(); render(); } };
  const downloadImages = () => {
    const table = find('Table')[0];
    const actions = table.props.columns.find(column => column.key === 'action').render(null, table.props.dataSource[0]);
    const dropdown = actions.props.children.find(child => child.type === 'Dropdown');
    dropdown.props.menu.items.find(item => item.key === 'images-zip').onClick({ domEvent: { stopPropagation() {} } });
  };
  render();
  return { settle, downloadImages, confirms, saved, errors, find };
}

test('总结图表 ZIP 已存在时只请求一次，直接保存相同响应', async () => {
  let requests = 0;
  const h = harness(async () => { requests++; return { data: new Blob(['zip']) }; });
  await h.settle(); h.downloadImages(); await h.settle();
  assert.equal(requests, 1);
  assert.deepEqual(h.saved, ['report_images.zip']);
  assert.equal(h.confirms.length, 0);
});

test('总结图表缺失先确认，生成成功后只重新请求一次', async () => {
  let requests = 0, generated = 0;
  const h = harness(async () => {
    if (++requests === 1) throw { response: { status: 404 } };
    return { data: new Blob(['zip']) };
  }, async () => generated++);
  await h.settle(); h.downloadImages(); await h.settle();
  assert.equal(requests, 1); assert.equal(generated, 0);
  await h.confirms[0].onOk(); await h.settle();
  assert.equal(requests, 2); assert.equal(generated, 1); assert.equal(h.saved.length, 1);
});

test('总结 ZIP 下载异常不伪造成功，不重复请求，显示中文', async () => {
  let requests = 0;
  const h = harness(async () => { requests++; throw { response: { status: 500 } }; });
  await h.settle(); h.downloadImages(); await h.settle();
  assert.equal(requests, 1); assert.equal(h.saved.length, 0); assert.equal(h.confirms.length, 0);
  assert.match(h.errors[0], /后端处理失败/);
});

test('总结选择排除明确没有指标的失败实验，兼容旧接口', async () => {
  const h = harness(async () => {});
  await h.settle();
  const group = h.find(undefined).find(node => Array.isArray(node.props?.options));
  assert.deepEqual(Array.from(group.props.options, item => item.value), ['healthy', 'legacy']);
});