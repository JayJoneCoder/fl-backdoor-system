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

function harness(download, generate = async () => {}, summaryData = [{ name: 'report', experiments: ['healthy'] }], experimentData = [{ name: 'healthy', has_metrics: true },
  { name: 'failed', has_metrics: false }, { name: 'legacy' }], remove = async () => {},
  detail = { name: 'report', summary_table: [], images: [], experiments_curve_data: {} }) {
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
    'Image', 'Row', 'Col', 'Spin', 'Divider', 'Dropdown', 'Pagination'].map(name => [name, name]));
  antd.Input = Object.assign('Input', { Search: 'Input.Search' });
  antd.Modal = Object.assign(() => {}, { confirm: value => confirms.push(value) });
  antd.message = { error: value => errors.push(value), success() {}, warning() {} };
  const icons = Object.fromEntries(['FileTextOutlined', 'PlusOutlined', 'BarChartOutlined', 'CodeOutlined',
    'DownloadOutlined', 'FileZipOutlined', 'FileImageOutlined', 'DeleteOutlined'].map(name => [name, name]));
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
      listExperiments: async () => ({ data: experimentData }),
      listSummaries: async () => ({ data: summaryData }),
      createSummary: async () => {}, generateSummaryPlots: generate, downloadSummaryZip: download,
      getSummaryDetail: async () => ({ data: detail }), deleteSummary: remove,
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
  const text = node => {
    if (typeof node === 'string' || typeof node === 'number') return String(node);
    if (!node || typeof node !== 'object') return '';
    if (Array.isArray(node)) return node.map(text).join('');
    return text(node.props?.children);
  };
  const settle = async () => { for (let i = 0; i < 6; i++) { await Promise.resolve(); render(); } };
  const downloadImages = () => {
    const table = find('Table')[0];
    const actions = table.props.columns.find(column => column.key === 'action').render(null, table.props.dataSource[0]);
    const dropdown = actions.props.children.find(child => child.type === 'Dropdown');
    dropdown.props.menu.items.find(item => item.key === 'images-zip').onClick({ domEvent: { stopPropagation() {} } });
  };
  const within = (value, type) => {
    const nodes = [];
    const walk = node => {
      if (!node || typeof node !== 'object') return;
      if (Array.isArray(node)) return node.forEach(walk);
      if (node.type === type) nodes.push(node);
      walk(node.props?.children);
    };
    walk(value); return nodes;
  };
  const allNodes = () => {
    const nodes = [];
    const walk = value => {
      if (!value || typeof value !== 'object') return;
      if (Array.isArray(value)) return value.forEach(walk);
      nodes.push(value);
      walk(value.props?.children);
    };
    walk(tree); return nodes;
  };
  const detailModal = () => allNodes().find(node =>
    typeof node.props?.title === 'string' && node.props.title.startsWith('总结详情'));
  const openDetail = async () => {
    const table = find('Table')[0];
    table.props.onRow(table.props.dataSource[0]).onClick();
    await settle();
  };
  const deleteRecord = () => {
    const table = find('Table')[0];
    const actions = table.props.columns.find(column => column.key === 'action').render(null, table.props.dataSource[0]);
    within(actions, 'Button').find(button => text(button) === '删除')
      .props.onClick({ stopPropagation() {} });
  };
  const deleteDetail = () => {
    within(detailModal().props.children, 'Button').find(button => text(button) === '删除本总结')
      .props.onClick();
  };
  render();
  return { settle, downloadImages, confirms, saved, errors, find, text, tree: () => tree, table: () => find('Table')[0],
    within, detailModal, openDetail, deleteRecord, deleteDetail };
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

test('总结列表默认每页 50 个，可按总结名称或包含的实验名称筛选', async () => {
  const summaries = [
    { name: 'alpha总结', experiments: ['exp-a'], created: null },
    { name: 'beta总结', experiments: ['exp-b'], created: null },
    { name: 'gamma总结', experiments: ['exp-c'], created: null },
  ];
  const h = harness(async () => {}, async () => {}, summaries);
  await h.settle();
  const table = h.table();
  assert.equal(table.props.pagination.pageSize, 50);
  assert.deepEqual([...table.props.pagination.pageSizeOptions], [20, 50, 100, 200]);
  assert.equal(table.props.dataSource.length, 3);

  const search = h.find('Input.Search')
    .find(node => node.props.placeholder === '搜索总结名称或包含的实验名称');
  search.props.onChange({ target: { value: 'exp-b' } });
  await h.settle();
  assert.deepEqual(h.table().props.dataSource.map(item => item.name), ['beta总结']);
  assert.equal(h.table().props.pagination.current, 1);

  search.props.onChange({ target: { value: 'GAMMA' } });
  await h.settle();
  assert.deepEqual(h.table().props.dataSource.map(item => item.name), ['gamma总结']);
});

test('总结列表删除需确认后调用删除接口', async () => {
  const removed = [];
  const h = harness(async () => {}, async () => {}, undefined, undefined, async (name) => { removed.push(name); });
  await h.settle();
  h.deleteRecord();
  await h.settle();
  assert.equal(removed.length, 0);
  assert.equal(h.confirms.length, 1);
  assert.equal(h.confirms[0].title, '删除总结');
  assert.match(h.confirms[0].content, /report/);
  await h.confirms[0].onOk();
  await h.settle();
  assert.deepEqual(removed, ['report']);
});

test('总结详情提供与列表一致的下载入口，并可直接删除当前总结', async () => {
  let requests = 0;
  const removed = [];
  const summaries = [{ name: 'report', experiments: ['healthy'], created: null }];
  const h = harness(async () => {
    requests++;
    return { data: new Blob(['zip']) };
  }, async () => {}, summaries, [{ name: 'healthy', has_metrics: true }],
  async (name) => { removed.push(name); summaries.splice(0, summaries.length); });
  await h.settle();
  await h.openDetail();

  const modal = h.detailModal();
  assert.equal(modal.props.open, true);
  const detailItems = h.within(modal.props.children, 'Dropdown')[0].props.menu.items;
  const actions = h.table().props.columns.find(column => column.key === 'action')
    .render(null, h.table().props.dataSource[0]);
  const listItems = actions.props.children.find(child => child.type === 'Dropdown').props.menu.items;
  assert.deepEqual(detailItems.map(item => item.key), listItems.map(item => item.key));

  detailItems.find(item => item.key === 'images-zip')
    .onClick({ domEvent: { stopPropagation() {} } });
  await h.settle();
  assert.equal(requests, 1);
  assert.deepEqual(h.saved, ['report_images.zip']);

  h.deleteDetail();
  await h.settle();
  assert.equal(removed.length, 0);
  assert.equal(h.confirms.at(-1).title, '删除总结');
  await h.confirms.at(-1).onOk();
  await h.settle();
  assert.deepEqual(removed, ['report']);
  assert.equal(h.detailModal().props.open, false);
});

test('新建总结的实验选择支持按实验名称或 UUID 搜索', async () => {
  const uuid = 'f7f8c75e-32a8-485b-ab2b-e0db3f1677ea';
  const experiments = [
    { name: '健康实验', run_id: uuid, has_metrics: true },
    { name: '失败实验', has_metrics: false },
    { name: '旧实验', run_id: 'legacy-id', has_metrics: true },
  ];
  const h = harness(async () => {}, async () => {}, [], experiments);
  await h.settle();
  const group = () => h.find(undefined).find(node => Array.isArray(node.props?.options));
  assert.deepEqual(Array.from(group().props.options, item => item.value), ['健康实验', '旧实验']);

  const search = h.find('Input.Search')
    .find(node => node.props.placeholder === '搜索实验名称或 UUID');
  search.props.onChange({ target: { value: uuid.slice(0, 8) } });
  await h.settle();
  assert.deepEqual(Array.from(group().props.options, item => item.value), ['健康实验']);

  search.props.onChange({ target: { value: '旧' } });
  await h.settle();
  assert.deepEqual(Array.from(group().props.options, item => item.value), ['旧实验']);
});
