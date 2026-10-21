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

const createHarness = (experiments, details = {}, overrides = {}) => {
  const slots = [], effects = [], errors = [], requests = [], downloads = [], confirms = [], opened = [], notices = [], copies = [];
  let cursor = 0, dirty = false, tree;
  const react = {
    useState: initial => {
      const index = cursor++;
      const slot = slots[index] ??= { value: initial };
      return [slot.value, value => { slot.value = typeof value === 'function' ? value(slot.value) : value; dirty = true; }];
    },
    useRef: initial => slots[cursor++] ??= { current: initial },
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
    visit(node.props?.actions, callback);
    visit(node.props?.extra, callback);
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
    runInNewContext(source, { exports, Blob,
      window: { URL: { createObjectURL: () => 'blob:download', revokeObjectURL() {} }, open: url => opened.push(url),
        navigator: { clipboard: { writeText: async value => { copies.push(value); } } } },
      document: { createElement: () => ({ click() { downloads.push(this.download); } }),
        body: { appendChild() {}, removeChild() {} } }, require: name => {
      assert.ok(name in imports, `Unexpected import: ${name}`);
      return imports[name];
    } });
    return exports;
  };
  const antd = Object.fromEntries(['Card', 'Row', 'Col', 'Spin', 'Button', 'Modal', 'Table',
    'Space', 'Divider', 'Image', 'Statistic', 'Descriptions', 'Tooltip', 'Dropdown', 'Tag', 'Alert',
    'Input', 'Pagination'].map(name => [name, name]));
  antd.Input = Object.assign('Input', { Search: 'Input.Search' });
  antd.Modal = Object.assign(() => {}, { confirm: value => confirms.push(value) });
  antd.Typography = { Text: 'Typography.Text' };
  antd.message = {
    error: value => { errors.push(typeof value === 'object' ? value.content : value); notices.push({ type: 'error', value }); },
    success: value => notices.push({ type: 'success', value }),
    loading: value => notices.push({ type: 'loading', value }),
  };
  const icons = Object.fromEntries(['BarChartOutlined', 'DownloadOutlined', 'FileTextOutlined',
    'ExperimentOutlined', 'AimOutlined', 'FileImageOutlined', 'FileZipOutlined', 'DeleteOutlined'].map(name => [name, name]));
  const charts = Object.fromEntries(['LineChart', 'Line', 'XAxis', 'YAxis', 'CartesianGrid',
    'Tooltip', 'Legend', 'ResponsiveContainer'].map(name => [name, `Recharts.${name}`]));
  const failOrReturn = value => {
    if (value instanceof Error || value?.response) throw value;
    return { data: value };
  };
  const listRequests = [];
  const api = {
    listExperiments: async params => {
      listRequests.push(params);
      return failOrReturn(typeof experiments === 'function' ? experiments(params) : experiments);
    },
    getExperimentDetail: async name => { requests.push(name); return failOrReturn(details[name]); },
    generatePlots: async () => {}, downloadExperimentImages: async () => {}, downloadExperimentAllFiles: async () => {},
    deleteExperiment: async () => {}, downloadExperimentFile: async () => ({ data: new Blob(['file']) }),
    ...overrides,
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
  return { settle, errors, requests, text, downloads, confirms, opened, notices, copies, listRequests,
    tree: () => tree,
    cards: () => findAll(tree, 'Card').filter(node => node.props.hoverable),
    modal: () => findAll(tree, antd.Modal)[0],
    dropdowns: () => findAll(tree, 'Dropdown'),
    pagination: () => findAll(tree, 'Pagination')[0],
    searchInput: () => findAll(tree, 'Input.Search')[0],
    nodes: (root, type) => findAll(root, type),
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

test('历史卡片显示实验者用户名，悬停显示 UID，点击复制 UID 且不打开详情', async () => {
  const h = createHarness([{ ...summary('新实验', uuid), owner: { uid: uuid, username: 'admin' } }]);
  await h.settle();
  const card = h.cards()[0];
  const label = h.texts(card).find(node => h.text(node) === '实验者：admin');
  assert.ok(label, '卡片显示实验者用户名');
  assert.ok(h.tooltips(card).some(node => node.props.title === `UID：${uuid}（点击复制）`));
  await label.props.onClick({ stopPropagation() {} });
  await h.settle();
  assert.deepEqual(h.copies, [uuid]);
  assert.match(h.notices.at(-1).value, /已复制实验者 UID/);
  assert.equal(h.requests.length, 0);
  assert.equal(h.modal().props.open, false);
});

test('详情显示实验者，点击用户名同样复制 UID', async () => {
  const h = createHarness([summary('新实验', uuid)],
    { 新实验: { ...detail('新实验', uuid), owner: { uid: uuid, username: 'admin' } } });
  await h.settle();
  await h.cards()[0].props.onClick();
  await h.settle();
  assert.ok(h.text(h.modal()).includes('admin'), '详情显示实验者用户名');
  const label = h.texts(h.modal()).find(node => h.text(node) === 'admin');
  assert.ok(label, '详情显示实验者用户名');
  await label.props.onClick();
  await h.settle();
  assert.deepEqual(h.copies, [uuid]);
  assert.match(h.notices.at(-1).value, /已复制实验者 UID/);
});

test('长实验名保留完整提示，卡片与指标布局使用统一样式', async () => {
  const name = 'very_long_federated_experiment_name_with_multiple_defenses_and_dataset_parameters';
  const h = createHarness([summary(name, uuid)]);
  await h.settle();
  const card = h.cards()[0];
  assert.equal(card.props.className, 'history-experiment-card');
  const title = h.nodes(card, 'div').find(node => node.props.className === 'history-experiment-card__name');
  assert.equal(title.props.title, name);
  assert.equal(h.text(title), name);
  assert.equal(h.nodes(card, 'Statistic').length, 2);
  assert.ok(h.nodes(card, 'Statistic').every(node => node.props.prefix === undefined));
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

test('失败且无指标的实验可打开详情，空值显示未记录，禁用图表但可查看日志', async () => {
  const run = { name: 'failed', run_id: uuid, status: 'failed', accuracy: null, asr: null,
    loss: null, last_round: null, can_plot: false, has_metrics: false, warnings: ['尚无指标 CSV'] };
  let plotted = 0;
  const h = createHarness([run], { failed: { ...run, error: '训练失败，请查看日志。', csv_files: [], images: [],
    metrics: { accuracy: null, asr: null, loss: null, last_round: null,
      curve_data: { round: [], accuracy: [], asr: [], loss: [] } } } }, { generatePlots: async () => plotted++ });
  await h.settle();
  assert.match(h.text(h.cards()[0]), /失败/);
  assert.ok(h.nodes(h.cards()[0], 'Statistic').every(node => node.props.value === '未记录'));
  await h.cards()[0].props.onClick(); await h.settle();
  const modal = h.modal();
  assert.equal(modal.props.open, true);
  assert.equal(h.nodes(modal, 'Recharts.LineChart').length, 0);
  const buttons = h.nodes(modal, 'Button');
  assert.equal(buttons.find(node => h.text(node) === '生成图表').props.disabled, true);
  assert.equal(buttons.find(node => h.text(node) === '主日志 CSV').props.disabled, true);
  buttons.find(node => h.text(node) === '查看运行日志').props.onClick();
  assert.equal(h.opened.at(-1), '/results/failed/run.log');
  assert.equal(plotted, 0);
  assert.ok(h.nodes(modal, 'Table')[0].props.dataSource.some(row => row.metric === 'accuracy' && row.value === '未记录'));
});

test('历史图表 ZIP 成功只请求一次，并直接保存收到的响应', async () => {
  let count = 0;
  const h = createHarness([summary('one', uuid)], {}, { downloadExperimentImages: async () => {
    count++; return { data: new Blob(['zip']) };
  } });
  await h.settle();
  h.dropdowns()[0].props.menu.items.find(item => item.key === 'images-zip').onClick({ domEvent: { stopPropagation() {} } });
  await h.settle();
  assert.equal(count, 1);
  assert.deepEqual(h.downloads, ['one_images.zip']);
  assert.equal(h.confirms.length, 0);
});

test('历史图表 ZIP 不存在时先确认，生成后只下载一次；无指标不提示生成', async () => {
  let count = 0, plotted = 0;
  const api = { downloadExperimentImages: async () => {
    if (++count === 1) throw { response: { status: 404 } };
    return { data: new Blob(['zip']) };
  }, generatePlots: async () => plotted++ };
  const h = createHarness([summary('one', uuid)], {}, api);
  await h.settle();
  h.dropdowns()[0].props.menu.items.find(item => item.key === 'images-zip').onClick({ domEvent: { stopPropagation() {} } });
  await h.settle();
  assert.equal(count, 1); assert.equal(plotted, 0);
  await h.confirms[0].onOk(); await h.settle();
  assert.equal(count, 2); assert.equal(plotted, 1);
  assert.equal(h.downloads.length, 1);
  const failed = createHarness([{ ...summary('failed', uuid), can_plot: false }], {}, {
    downloadExperimentImages: async () => { throw { response: { status: 404 } }; },
  });
  await failed.settle();
  failed.dropdowns()[0].props.menu.items.find(item => item.key === 'images-zip').onClick({ domEvent: { stopPropagation() {} } });
  await failed.settle();
  assert.equal(failed.confirms.length, 0);
  assert.match(failed.errors[0], /指标不完整/);
});

const click = button => button.props.onClick({ stopPropagation() {} });
const button = (h, root, label) => h.nodes(root, 'Button').find(node => h.text(node) === label);

test('删除是卡片的第三个操作，取消不请求，确认携带 UUID 并移除卡片和已打开的详情', async () => {
  const deleted = [];
  const h = createHarness([summary('one', uuid)], { one: detail('one', uuid) }, {
    deleteExperiment: async (...args) => deleted.push(args),
  });
  await h.settle();
  assert.equal(h.text(h.cards()[0].props.actions[2]), '删除');
  click(button(h, h.cards()[0], '删除'));
  assert.equal(deleted.length, 0);
  assert.match(h.confirms[0].content, /无法撤销/);
  assert.equal(h.confirms[0].okButtonProps.danger, true);
  await h.cards()[0].props.onClick(); await h.settle();
  await h.confirms[0].onOk(); await h.settle();
  assert.deepEqual(deleted, [['one', uuid]]);
  assert.equal(h.cards().length, 0);
  assert.equal(h.modal().props.open, false);
});

test('删除失败保留原卡片和详情并显示中文原因，活动实验禁用删除', async () => {
  const h = createHarness([summary('one', uuid)], { one: detail('one', uuid) }, {
    deleteExperiment: async () => { throw { response: { status: 409, data: { detail: '该实验正在用于生成总结，请等待完成后再删除。' } } }; },
  });
  await h.settle(); await h.cards()[0].props.onClick(); await h.settle();
  click(button(h, h.cards()[0], '删除'));
  await assert.rejects(h.confirms[0].onOk()); await h.settle();
  assert.equal(h.cards().length, 1);
  assert.equal(h.modal().props.open, true);
  assert.match(h.errors[0], /生成总结/);
  assert.equal(button(h, h.cards()[0], '删除').props.loading, false);
  const active = createHarness(['starting', 'running', 'stopping', 'cleaning', 'cleanup_failed'].map(status => ({ ...summary(status, uuid), status })));
  await active.settle();
  assert.ok(active.cards().every(card => button(active, card, '删除').props.disabled));
});

test('生成图表立即提示和加载，防重复点击并禁止删除，完成后解除', async () => {
  let finish, count = 0;
  const h = createHarness([summary('one', uuid)], {}, {
    generatePlots: () => { count++; return new Promise(resolve => { finish = resolve; }); },
  });
  await h.settle();
  const trigger = button(h, h.cards()[0], '生成图表');
  const first = click(trigger);
  await click(trigger); await h.settle();
  assert.equal(count, 1);
  assert.match(h.notices[0].value.content, /正在生成图表，请稍候/);
  assert.equal(h.notices[0].value.duration, 0);
  assert.equal(button(h, h.cards()[0], '生成中…').props.loading, true);
  assert.equal(button(h, h.cards()[0], '删除').props.disabled, true);
  finish(); await first; await h.settle();
  assert.equal(button(h, h.cards()[0], '生成图表').props.loading, false);
  assert.equal(button(h, h.cards()[0], '删除').props.disabled, false);
  assert.equal(h.notices.at(-1).value.key, h.notices[0].value.key);
});

test('生成失败替换加载提示，按钮恢复可用；不同实验各自维护加载状态', async () => {
  const resolvers = {};
  const h = createHarness([summary('one', uuid), summary('two', uuid)], {}, {
    generatePlots: name => new Promise((resolve, reject) => { resolvers[name] = { resolve, reject }; }),
  });
  await h.settle();
  const one = click(button(h, h.cards()[0], '生成图表'));
  const two = click(button(h, h.cards()[1], '生成图表'));
  await h.settle();
  assert.ok(h.cards().every(card => button(h, card, '生成中…').props.loading));
  resolvers.one.reject(new Error('private traceback')); await one; await h.settle();
  assert.match(h.errors[0], /图表生成失败/);
  assert.equal(button(h, h.cards()[0], '生成图表').props.loading, false);
  assert.equal(button(h, h.cards()[1], '生成中…').props.loading, true);
  resolvers.two.resolve(); await two; await h.settle();
});

test('详情下载功能完整，三类 CSV、日志和两类 ZIP 复用卡片操作且各请求一次', async () => {
  const files = [], zip = [];
  const data = { ...detail('one', uuid), csv_files: ['one.csv', 'one_clients.csv', 'one_metrics.csv'] };
  const h = createHarness([summary('one', uuid)], { one: data }, {
    downloadExperimentFile: async (...args) => { files.push(args); return { data: new Blob(['file']) }; },
    downloadExperimentImages: async name => { zip.push(['images', name]); return { data: new Blob(['zip']) }; },
    downloadExperimentAllFiles: async name => { zip.push(['all', name]); return { data: new Blob(['zip']) }; },
  });
  await h.settle(); await h.cards()[0].props.onClick(); await h.settle();
  assert.equal(h.nodes(h.modal(), 'Dropdown').length, 0);
  for (const label of ['主日志 CSV', '客户端日志 CSV', '检测指标 CSV', '下载运行日志', '下载图表 (ZIP)', '下载全部文件 (ZIP)']) {
    await click(button(h, h.modal(), label));
    await h.settle();
  }
  assert.deepEqual(files, [['one', 'one.csv'], ['one', 'one_clients.csv'], ['one', 'one_metrics.csv'], ['one', 'run.log']]);
  assert.deepEqual(zip, [['images', 'one'], ['all', 'one']]);
  assert.deepEqual(h.downloads, ['one.csv', 'one_clients.csv', 'one_metrics.csv', 'run.log', 'one_images.zip', 'one_all_files.zip']);
});

test('单文件下载失败给出中文错误，不打开空页面或伪造下载', async () => {
  const h = createHarness([summary('one', uuid)], { one: detail('one', uuid) }, {
    downloadExperimentFile: async () => { throw { response: { status: 404 } }; },
  });
  await h.settle(); await h.cards()[0].props.onClick(); await h.settle();
  await click(button(h, h.modal(), '下载运行日志'));
  await h.settle();
  assert.match(h.errors[0], /不存在/);
  assert.equal(h.downloads.length, 0);
  assert.equal(h.opened.length, 0);
});

test('历史记录默认每页 100 个，可调整每页数量并按偏移量查询', async () => {
  const h = createHarness(() => ({ items: [summary('one', uuid)], total: 141 }));
  await h.settle();
  assert.deepEqual({ ...h.listRequests[0] }, { search: '', limit: 100, offset: 0, paginated: true });
  assert.equal(h.pagination().props.pageSize, 100);
  assert.deepEqual([...h.pagination().props.pageSizeOptions], [20, 50, 100, 200]);
  h.pagination().props.onChange(2, 100);
  await h.settle();
  assert.deepEqual({ ...h.listRequests.at(-1) }, { search: '', limit: 100, offset: 100, paginated: true });
  assert.equal(h.pagination().props.current, 2);
  h.pagination().props.onChange(3, 50);
  await h.settle();
  assert.deepEqual({ ...h.listRequests.at(-1) }, { search: '', limit: 50, offset: 0, paginated: true });
  assert.equal(h.pagination().props.current, 1);
});

test('历史记录可按实验名称或 UUID 搜索，清空后恢复全部', async () => {
  const h = createHarness(() => ({ items: [], total: 0 }));
  await h.settle();
  const input = h.searchInput();
  assert.equal(input.props.placeholder, '搜索实验名称或 UUID');
  input.props.onChange({ target: { value: `  ${uuid}  ` } });
  input.props.onSearch(`  ${uuid}  `);
  await h.settle();
  assert.equal(h.listRequests.at(-1).search, uuid);
  assert.equal(h.listRequests.at(-1).offset, 0);
  assert.match(h.text(h.tree()), /没有匹配的实验/);
  input.props.onChange({ target: { value: '' } });
  await h.settle();
  assert.equal(h.listRequests.at(-1).search, '');
  assert.match(h.text(h.tree()), /暂无实验记录/);
});
