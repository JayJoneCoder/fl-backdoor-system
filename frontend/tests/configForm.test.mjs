import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compile = file => ts.transpileModule(readFileSync(new URL(file, import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const code = compile('../src/components/ConfigForm/index.tsx');
const requestCode = compile('../src/api/checkedFetch.ts');

// Execute the real component with deterministic hooks/Form/Modal and requests.
const harness = async ({ extraFields = {}, extraConfig = {} } = {}) => {
  const slots = [], effects = [], dialogs = [], successes = [], errors = [], calls = [];
  let cursor = 0, dirty = false, tree, apiFailure = null, formFailure = null;
  let fetchResponse = () => new Response('{}', { status: 200 });
  const schema = {
    'batch-size': { type: 'integer', default: 32, group: 'federated' },
    defense: { type: 'string', default: 'none', group: 'defense' },
    ...extraFields,
  };
  let saved = { 'batch-size': 32, defense: 'none', ...extraConfig };
  let values = { ...saved };
  const ref = {};
  const react = {
    useState: initial => {
      const index = cursor++;
      const slot = slots[index] ??= { value: initial };
      return [slot.value, update => {
        const next = typeof update === 'function' ? update(slot.value) : update;
        if (!Object.is(next, slot.value)) { slot.value = next; dirty = true; }
      }];
    },
    useEffect: (effect, deps) => {
      const index = cursor++;
      const slot = slots[index];
      if (!slot || !deps || !slot.deps || deps.some((value, i) => !Object.is(value, slot.deps[i]))) {
        slots[index] = { deps }; effects.push(effect);
      }
    },
    useImperativeHandle: (target, create) => { target.current = create(); },
    forwardRef: component => component,
  };
  const form = {
    setFieldsValue: update => { values = { ...values, ...update }; },
    getFieldsValue: () => ({ ...values }),
    getFieldValue: name => values[name],
    setFieldValue: (name, value) => { values[name] = value; },
    resetFields: () => { values = { ...saved }; },
    validateFields: async () => { if (formFailure) throw formFailure; return { ...values }; },
  };
  const Form = { useForm: () => [form], Item: 'FormItem' };
  const Input = { TextArea: 'TextArea' };
  const Select = { Option: 'Option' };
  const antd = Object.fromEntries(['InputNumber', 'Switch', 'Button', 'Space', 'Card', 'Row', 'Col', 'Tooltip', 'Divider', 'Upload', 'Alert', 'Dropdown', 'Table', 'Tag'].map(name => [name, name]));
  Object.assign(antd, { Form, Input, Select,
    Modal: { confirm: options => { dialogs.push(options); }, destroyAll() {} },
    message: { success: text => successes.push(text), error: text => errors.push(text) },
  });
  const api = {
    getConfigSchema: async () => ({ data: { fields: schema, groups: ['federated', 'defense'] } }),
    getConfig: async () => ({ data: { ...saved } }),
    updateConfig: async (update, backupName) => {
      calls.push({ type: 'save', update, backupName });
      if (apiFailure) throw apiFailure;
      saved = { ...saved, ...update };
    },
  };
  const jsx = (type, props, key) => ({ type, props, key });
  const walk = (node, callback) => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { node.forEach(child => walk(child, callback)); return; }
    callback(node);
    walk(node.props?.children, callback);
    walk(node.props?.footer, callback);
  };
  const exports = {};
  const requestExports = {};
  const fetch = async (url, options) => {
    calls.push({ type: 'fetch', url, method: options?.method, body: options?.body });
    return fetchResponse(url, options);
  };
  runInNewContext(requestCode, { exports: requestExports, fetch, Error, Headers });
  runInNewContext(code, {
    exports,
    require: name => {
      if (name === 'react') return react;
      if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: 'Fragment' };
      if (name === 'antd') return antd;
      if (name === '@ant-design/icons') return {};
      if (name === '../../api/client') return api;
      if (name === '../../api/endpoints') return { getBackendUrl: path => path };
      if (name === '../../api/checkedFetch') return requestExports;
      if (name === '../AttackPreview') return { default: 'AttackPreview' };
      throw new Error(`Unexpected import ${name}`);
    },
    document: { getElementById: () => ({ value: 'before-change' }) },
    fetch, URL, Blob, console, Error,
  });
  const render = () => {
    do {
      dirty = false; cursor = 0;
      tree = exports.default({}, ref);
      effects.splice(0).forEach(effect => effect());
    } while (dirty);
  };
  const settle = async () => { for (let i = 0; i < 6; i++) await Promise.resolve(); render(); };
  const find = predicate => { let result; walk(tree, node => { if (predicate(node)) result = node; }); return result; };
  render(); await settle();
  return {
    ref, dialogs, successes, errors, calls, settle,
    failSave: error => { apiFailure = error; },
    failFormValidation: error => { formFailure = error; },
    respondFetch: callback => { fetchResponse = callback; },
    edit: update => {
      values = { ...values, ...update };
      find(node => node.type === Form).props.onValuesChange(update, values);
      render();
    },
    submit: async () => { await find(node => node.type === Form).props.onFinish(values); await settle(); },
    gate: () => ref.current.checkUnsavedBeforeStart(),
    clickButton: async label => { await find(node => node.type === 'Button' && node.props.children === label).props.onClick(); },
    importFile: file => find(node => node.type === 'Upload' && node.props.beforeUpload)?.props.beforeUpload(file),
    deleteBackup: filename => {
      const table = find(node => node.type === 'Table' && node.props.rowKey === 'filename');
      const row = table.props.columns.find(column => column.key === 'action').render(null, { filename });
      let button;
      walk(row, node => { if (node.type === 'Button' && node.props.children === '删除') button = node; });
      button.props.onClick();
    },
    setRaw: text => { find(node => node.type === 'TextArea' && node.props.placeholder === '加载中...').props.onChange({ target: { value: text } }); render(); },
    defenseKey: () => find(node => node.type === 'Card' && node.key === 'defense')?.key,
    section: group => find(node => node.type === 'Card' && node.key === group),
    sectionTag: group => {
      let tag;
      walk(find(node => node.type === 'Card' && node.key === group)?.props.title, node => {
        if (node.type === 'Tag') tag = node;
      });
      return tag;
    },
    field: name => find(node => node.type === 'FormItem' && node.props.name === name),
    formClass: () => find(node => node.type === Form).props.className,
    toggleSection: group => {
      let toggle;
      walk(find(node => node.type === 'Card' && node.key === group)?.props.extra, node => {
        if (node.type === 'Button') toggle = node;
      });
      toggle.props.onClick(); render();
    },
    advancedOpen: () => find(node => node.type === antd.Modal && node.props.title === '高级配置模式（直接编辑 pyproject.toml）')?.props.open,
  };
};

test('保存失败的启动检查返回取消，不会误报成功', async () => {
  for (const button of ['onOk', 'onCancel']) {
    const h = await harness();
    h.edit({ 'batch-size': -1 });
    h.failSave({ response: { data: { detail: 'batch-size: must be a positive integer.' } } });
    const outcome = h.gate();
    await h.dialogs.at(-1)[button]();
    assert.equal((await outcome).action, 'cancel');
    assert.equal(h.successes.length, 0);
    assert.ok(h.errors.some(text => /批次大小.*必须为正整数/.test(text)));
    assert.equal(h.calls.filter(call => call.type === 'save').length, 1);
  }
});

test('成功保存后才允许启动，备份要求在同一保存请求中传递', async () => {
  for (const button of ['onOk', 'onCancel']) {
    const h = await harness();
    h.edit({ 'batch-size': 64 });
    const outcome = h.gate();
    await h.dialogs.at(-1)[button]();
    assert.equal((await outcome).action, button === 'onOk' ? 'backup' : 'save');
    const saves = h.calls.filter(call => call.type === 'save');
    assert.equal(saves.length, 1);
    assert.equal(saves[0].update['batch-size'], 64);
    assert.equal(saves[0].backupName, button === 'onOk' ? '' : undefined);
    assert.equal(h.calls.filter(call => call.type === 'fetch').length, 0);
  }
});

test('表单校验失败时不会请求保存，也不会允许启动', async () => {
  const h = await harness();
  h.edit({ 'batch-size': 64 });
  h.failFormValidation(new Error('form invalid'));
  const outcome = h.gate();
  await h.dialogs.at(-1).onOk();
  assert.equal((await outcome).action, 'cancel');
  assert.equal(h.calls.filter(call => call.type === 'save').length, 0);
});

test('普通表单保存失败不产生成功提示，保留未保存状态', async () => {
  const h = await harness();
  h.edit({ 'batch-size': -1 });
  h.failSave(new Error('invalid settings'));
  await h.submit();
  assert.equal(h.successes.length, 0);
  assert.ok(h.errors.some(text => text.includes('配置操作失败')));
  assert.ok(h.errors.every(text => !text.includes('invalid settings')));
  const outcome = h.gate();
  await h.dialogs.at(-1).onCancel();
  assert.equal((await outcome).action, 'cancel');
});

test('高级保存的 400 响应不关闭编辑器、不显示成功', async () => {
  const h = await harness();
  h.respondFetch((url, options) => new Response(JSON.stringify(options?.method === 'POST'
    ? { detail: 'Invalid TOML' } : { content: 'old config' }), { status: options?.method === 'POST' ? 400 : 200 }));
  await h.clickButton('高级模式');
  await h.dialogs.at(-1).onOk(); await h.settle();
  assert.equal(h.advancedOpen(), true);
  h.setRaw('bad [');
  await h.clickButton('保存并应用...');
  await assert.rejects(h.dialogs.at(-1).onOk(), /Invalid TOML/);
  await h.settle();
  assert.equal(h.advancedOpen(), true);
  assert.equal(h.successes.length, 0);
  assert.ok(h.errors.some(text => text.includes('TOML 配置格式不正确')));
  const post = h.calls.filter(call => call.type === 'fetch' && call.method === 'POST');
  assert.equal(post.length, 1);
  assert.equal(post[0].url, '/api/config/raw');
  assert.equal(JSON.parse(post[0].body).backup_current, true);
});

test('高级保存备份旧配置与应用新配置只用一个请求', async () => {
  const h = await harness();
  h.respondFetch((url, options) => new Response(JSON.stringify(options?.method === 'POST'
    ? { status: 'success' } : url.endsWith('/raw') ? { content: 'old config' } : { backups: [] }), { status: 200 }));
  await h.clickButton('高级模式');
  await h.dialogs.at(-1).onOk(); await h.settle();
  h.setRaw('new config');
  await h.clickButton('保存并应用...');
  await h.dialogs.at(-1).onOk(); await h.settle();
  const posts = h.calls.filter(call => call.type === 'fetch' && call.method === 'POST');
  assert.equal(posts.length, 1);
  assert.deepEqual(JSON.parse(posts[0].body), { content: 'new config', backup_current: true, backup_name: 'before-change' });
  assert.equal(h.advancedOpen(), false);
  assert.ok(h.successes.some(text => text.includes('旧配置已备份')));
});

test('防御配置组有稳定 key', async () => {
  const h = await harness();
  assert.equal(h.defenseKey(), 'defense');
});

test('脏字段使用独立样式与红色分组提示，还原原值后清除且不冒充校验错误', async () => {
  const h = await harness();
  assert.equal(h.formClass(), 'configuration-form');
  assert.equal(h.field('batch-size').props.className, undefined);
  assert.equal(h.field('batch-size').props.validateStatus, undefined);
  assert.equal(h.sectionTag('federated').props.color, 'default');
  h.edit({ 'batch-size': 64 });
  assert.equal(h.field('batch-size').props.className, 'is-unsaved');
  assert.equal(h.field('batch-size').props.validateStatus, undefined);
  assert.equal(h.sectionTag('federated').props.color, 'error');
  assert.equal(h.sectionTag('federated').props.children, '有未保存的更改');
  h.edit({ 'batch-size': 32 });
  assert.equal(h.field('batch-size').props.className, undefined);
  assert.equal(h.field('batch-size').props.validateStatus, undefined);
  assert.equal(h.sectionTag('federated').props.color, 'default');
  assert.equal(h.sectionTag('federated').props.children, '所有更改已保存');
  assert.equal(h.calls.filter(call => call.type === 'save').length, 0);
});

test('防御字段与普通字段均显示红色提示，保存成功后清除', async () => {
  const h = await harness();
  h.edit({ 'batch-size': 64, defense: 'krum' });
  for (const field of ['batch-size', 'defense']) {
    assert.equal(h.field(field).props.className, 'is-unsaved');
    assert.equal(h.field(field).props.validateStatus, undefined);
  }
  for (const group of ['federated', 'defense']) {
    assert.equal(h.sectionTag(group).props.color, 'error');
  }
  await h.submit();
  for (const field of ['batch-size', 'defense']) {
    assert.equal(h.field(field).props.className, undefined);
    assert.equal(h.field(field).props.validateStatus, undefined);
  }
  for (const group of ['federated', 'defense']) {
    assert.equal(h.sectionTag(group).props.color, 'default');
  }
});

test('保存失败保留脏字段红色边框与未保存红色标签', async () => {
  const h = await harness();
  h.edit({ 'batch-size': 64 });
  h.failSave(new Error('save failed'));
  await h.submit();
  assert.equal(h.field('batch-size').props.validateStatus, undefined);
  assert.equal(h.field('batch-size').props.className, 'is-unsaved');
  assert.equal(h.sectionTag('federated').props.color, 'error');
});

test('自定义字段的未保存红色提示与标准字段一致，还原后清除', async () => {
  const h = await harness({ extraConfig: { 'custom-setting': 'original' } });
  assert.equal(h.sectionTag('other').props.color, 'default');
  h.edit({ 'custom-setting': 'updated' });
  assert.equal(h.field('custom-setting').props.className, 'is-unsaved');
  assert.equal(h.field('custom-setting').props.validateStatus, undefined);
  assert.equal(h.sectionTag('other').props.color, 'error');
  assert.equal(h.sectionTag('other').props.children, '有未保存的更改');
  h.edit({ 'custom-setting': 'original' });
  assert.equal(h.field('custom-setting').props.className, undefined);
  assert.equal(h.sectionTag('other').props.color, 'default');
});

test('防御组各子区域均使用独立未保存样式，保存后清除', async () => {
  const fields = [
    'client-defense-filter-ratio',
    'detection-z-threshold',
    'defense-krum-f',
  ];
  const h = await harness({
    extraFields: Object.fromEntries(fields.map(key => [key, { type: 'float', group: 'defense' }])),
    extraConfig: Object.fromEntries(fields.map(key => [key, 1])),
  });
  h.edit(Object.fromEntries(fields.map(key => [key, 2])));
  for (const key of fields) {
    assert.equal(h.field(key).props.className, 'is-unsaved');
    assert.equal(h.field(key).props.validateStatus, undefined);
  }
  assert.equal(h.sectionTag('defense').props.color, 'error');
  await h.submit();
  for (const key of fields) assert.equal(h.field(key).props.className, undefined);
  assert.equal(h.sectionTag('defense').props.color, 'default');
});

test('未保存样式覆盖输入框、数字框与选择框的红色边框及聚焦状态', () => {
  const css = readFileSync(new URL('../src/App.css', import.meta.url), 'utf8');
  assert.match(css, /\.configuration-form \.is-unsaved \.ant-form-item-control-input-content > :is\(\.ant-input, \.ant-input-affix-wrapper, \.ant-input-number, \.ant-select\)\s*\{\s*border-color: var\(--ant-color-error, #c74949\);/);
  assert.match(css, /:is\(\.ant-input:focus, \.ant-input-affix-wrapper:focus-within, \.ant-input-number:focus-within, \.ant-select:focus-within\)\s*\{[^}]*box-shadow: 0 0 0 2px var\(--ant-color-error-outline,/);
  assert.match(css, /\.is-unsaved \.ant-form-item-label > label::after\s*\{[^}]*background: var\(--ant-color-error, #c74949\);/);
});

test('配置组使用随容器宽度调整的字段网格，而不是固定三列', async () => {
  const h = await harness();
  for (const group of ['federated', 'defense']) {
    const children = h.section(group).props.children;
    const grids = Array.isArray(children) ? children : [children];
    const grid = grids.find(node => node?.type === 'Row');
    assert.equal(grid.props.className, 'config-field-grid');
  }
});

test('配置区域收起与展开保留表单内容和未保存值，不发送保存请求', async () => {
  const h = await harness();
  h.edit({ 'batch-size': 71 });
  for (const group of ['federated', 'defense']) {
    assert.notEqual(h.section(group).props.styles.body.display, 'none');
    const fields = h.section(group).props.children;
    h.toggleSection(group);
    assert.equal(h.section(group).props.styles.body.display, 'none');
    assert.ok(h.section(group).props.children);
    assert.deepEqual(h.section(group).props.children, fields);
    h.toggleSection(group);
    assert.notEqual(h.section(group).props.styles.body.display, 'none');
  }
  assert.equal(h.calls.filter(call => call.type === 'save').length, 0);
  await h.submit();
  assert.equal(h.calls.find(call => call.type === 'save').update['batch-size'], 71);
});

test('备份重名必须等待用户选择，重试失败不继续导入或误报成功', async () => {
  const h = await harness();
  let attempts = 0;
  h.respondFetch(() => new Response(JSON.stringify({ detail: 'backup failed' }), { status: ++attempts === 1 ? 409 : 500 }));
  h.importFile({ name: 'config.toml' });
  const importDialog = h.dialogs.at(-1);
  let finished = false;
  const importing = importDialog.onOk().then(() => { finished = true; });
  await h.settle();
  const conflictDialog = h.dialogs.at(-1);
  assert.equal(conflictDialog.title, '文件已存在');
  assert.equal(finished, false);
  await assert.rejects(conflictDialog.onOk(), /backup failed/);
  await importing;
  assert.ok(h.errors.some(text => text.includes('备份失败，取消导入')));
  assert.equal(h.successes.length, 0);
});

test('删除失败会显示后端错误，不显示删除成功', async () => {
  const h = await harness();
  h.respondFetch(() => new Response(JSON.stringify({ detail: 'cannot delete' }), { status: 500 }));
  h.deleteBackup('saved.toml');
  await h.dialogs.at(-1).onOk();
  assert.equal(h.successes.length, 0);
  assert.ok(h.errors.some(text => text.includes('后端处理失败（HTTP 500）')));
  assert.ok(h.errors.every(text => !text.includes('cannot delete')));
});

test('草稿备份失败不会显示创建成功', async () => {
  const h = await harness();
  h.respondFetch(() => new Response(JSON.stringify({ detail: 'invalid draft' }), { status: 400 }));
  await h.clickButton('仅创建备份');
  await assert.rejects(h.dialogs.at(-1).onOk(), /invalid draft/);
  assert.equal(h.successes.length, 0);
});