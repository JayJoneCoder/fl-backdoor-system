import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const code = ts.transpileModule(readFileSync(new URL('../src/components/AttackPreview/index.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const errorHelpers = {};
runInNewContext(ts.transpileModule(readFileSync(new URL('../src/api/checkedFetch.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023 },
}).outputText, { exports: errorHelpers });

function harness(initial = {}, responseFactory) {
  const slots = [], effects = [], timers = new Map(), requests = [];
  let cursor = 0, dirty = false, nextTimer = 0, tree;
  let config = { dataset: 'mnist', attack: 'dba', 'num-clients': 10, 'target-label': 0, 'poison-rate': 0, ...initial };
  const react = {
    useState(initialValue) {
      const index = cursor++;
      const slot = slots[index] ??= { value: initialValue };
      return [slot.value, update => {
        const value = typeof update === 'function' ? update(slot.value) : update;
        if (!Object.is(value, slot.value)) { slot.value = value; dirty = true; }
      }];
    },
    useEffect(effect, deps) {
      const index = cursor++;
      const old = slots[index];
      if (!old || deps.some((value, i) => !Object.is(value, old.deps[i]))) {
        old?.cleanup?.();
        const slot = slots[index] = { deps };
        effects.push(() => { slot.cleanup = effect(); });
      }
    },
  };
  const jsx = (type, props) => ({ type, props });
  const exports = {};
  runInNewContext(code, {
    exports, AbortController,
    window: { setTimeout: callback => { const id = ++nextTimer; timers.set(id, callback); return id; }, clearTimeout: id => timers.delete(id) },
    require(name) {
      if (name === 'react') return react;
      if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: 'Fragment' };
      if (name === 'antd') return { ...Object.fromEntries(['Alert', 'Button', 'Card', 'Col', 'InputNumber', 'Row', 'Select', 'Space', 'Spin', 'Tooltip'].map(n => [n, n])), Typography: { Text: 'Text', Title: 'Title' } };
      if (name === '@ant-design/icons') return {};
      if (name === '../../api/endpoints') return { getBackendUrl: path => path };
      if (name === '../../api/checkedFetch') return {
        ...errorHelpers,
        checkedFetch: async (url, options) => {
          const body = JSON.parse(options.body);
          requests.push({ url, body, signal: options.signal });
          if (responseFactory) return responseFactory(body);
          return { json: async () => ({ ...body, total: 100, label_text: '5', target_label_text: '0', changed_pixels: 4,
            clean_image: `clean-${body.index}`, poisoned_image: `poisoned-${body.index}`,
            ...(['dba', 'fcba'].includes(body.config.attack) ? {
              local_image: `local-${body.index}-${body.client_id}`, global_image: `global-${body.index}`,
              local_changed_pixels: 4, global_changed_pixels: 16, local_trigger_index: body.client_id,
              local_trigger_count: body.config.attack === 'fcba' ? 14 : 4, local_sub_blocks: 1,
            } : {}),
          }) };
        },
      };
      throw Error(`Unexpected import ${name}`);
    },
  });
  function render() {
    do {
      dirty = false; cursor = 0;
      tree = exports.default({ config, ready: true });
      effects.splice(0).forEach(effect => effect());
    } while (dirty);
  }
  function walk(node, visit) {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { node.forEach(child => walk(child, visit)); return; }
    visit(node); walk(node.props?.children, visit); walk(node.props?.extra, visit);
  }
  function find(predicate) { let result; walk(tree, node => { if (predicate(node)) result = node; }); return result; }
  function findAll(predicate) { const results = []; walk(tree, node => { if (predicate(node)) results.push(node); }); return results; }
  const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); render(); };
  render();
  return {
    requests, find, findAll, settle,
    async flush() { const batch = [...timers.values()]; timers.clear(); batch.forEach(cb => cb()); await settle(); },
    edit(update) { config = { ...config, ...update }; render(); },
    choose(index) { find(n => n.type === 'InputNumber' && n.props['aria-label'] === '训练集图片编号').props.onChange(index); render(); },
    click(label) { find(n => n.type === 'Button' && n.props.children === label).props.onClick(); render(); },
    changeClient(value) { find(n => n.type === 'InputNumber' && n.props['aria-label'] === '预览客户端 ID').props.onChange(value); render(); },
    unmount() { slots.forEach(slot => slot.cleanup?.()); },
  };
}

test('默认展示第一张，预览为只读 POST 请求并标注强制植入示意', async () => {
  const h = harness();
  await h.flush();
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].url, '/api/attack/preview');
  assert.equal(h.requests[0].body.index, 0);
  assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('原始')).props.src, 'clean-0');
  assert.ok(h.find(n => n.type === 'Alert' && n.props.title.includes('不代表')));
});

test('按编号及上一张下一张选择图片，切换数据集回到第一张', async () => {
  const h = harness(); await h.flush();
  h.choose(25); await h.flush();
  assert.equal(h.requests.at(-1).body.index, 24);
  h.click('下一张'); await h.flush();
  assert.equal(h.requests.at(-1).body.index, 25);
  h.click('上一张'); await h.flush();
  assert.equal(h.requests.at(-1).body.index, 24);
  h.edit({ dataset: 'cifar10' }); await h.flush();
  assert.equal(h.requests.at(-1).body.index, 0);
});

test('DBA／FCBA 同时展示原图、指定客户端训练投毒图和完整触发器，投毒率为零仍展示', async () => {
  for (const attack of ['dba', 'fcba']) {
    const h = harness({ attack }); await h.flush();
    assert.equal(h.findAll(n => n.type === 'img').length, 3);
    assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('局部')).props.src, 'local-0-0');
    assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('完整')).props.src, 'global-0');
    assert.equal(h.find(n => n.type === 'Select'), undefined);
    assert.ok(h.find(n => n.type === 'Alert' && n.props.description.includes('假设所选客户端为恶意客户端')));
    h.changeClient(4); await h.flush();
    assert.equal(h.requests.at(-1).body.scope, 'local');
    assert.equal(h.requests.at(-1).body.client_id, 4);
    assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('局部')).props.src, 'local-0-4');
    assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('完整')).props.src, 'global-0');
    h.edit({ 'num-clients': 2 }); await h.flush();
    assert.equal(h.requests.at(-1).body.client_id, 1);
  }
});

test('非分布式攻击保留两图，切换攻击后不残留局部或完整触发器', async () => {
  const h = harness(); await h.flush();
  assert.equal(h.findAll(n => n.type === 'img').length, 3);
  for (const attack of ['badnets', 'wanet', 'frequency', 'none']) {
    h.edit({ attack }); await h.flush();
    assert.equal(h.findAll(n => n.type === 'img').length, 2);
    assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('局部')), undefined);
    assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('完整')), undefined);
    assert.equal(h.find(n => n.props?.['aria-label'] === '预览客户端 ID'), undefined);
  }
  h.edit({ attack: 'fcba' }); await h.flush();
  assert.equal(h.findAll(n => n.type === 'img').length, 3);
});

test('FCBA 组合子块数量和两种变化像素数分别展示', async () => {
  const h = harness({ attack: 'fcba' }, async body => ({ json: async () => ({
    ...body, total: 100, label_text: '5', target_label_text: '0', clean_image: 'clean',
    local_image: 'combination', global_image: 'full', local_trigger_index: 4, local_trigger_count: 14,
    local_sub_blocks: 2, local_changed_pixels: 8, global_changed_pixels: 16,
  }) }));
  await h.flush();
  assert.ok(h.find(n => n.type === 'div' && JSON.stringify(n.props.children).includes('2 个子块')));
  const changes = h.findAll(n => n.type === 'Text' && Array.isArray(n.props.children) && n.props.children[0] === '变化像素：');
  assert.deepEqual(changes.map(n => n.props.children[1]), [8, 16]);
});

test('未保存对抗性安全测试参数更新触发新预览，修改实验名和防御不触发请求', async () => {
  const h = harness(); await h.flush();
  h.edit({ 'trigger-size': 6 }); await h.flush();
  assert.equal(h.requests.at(-1).body.config['trigger-size'], 6);
  const count = h.requests.length;
  h.edit({ 'run-name': 'draft', defense: 'krum' }); await h.flush();
  assert.equal(h.requests.length, count);
});

test('参数快速变更合并请求，旧响应和卸载后的响应不会覆盖新图', async () => {
  const pending = [];
  const h = harness({}, body => new Promise(resolve => {
    pending.push(() => resolve({ json: async () => ({
      ...body, total: 100, clean_image: `clean-${body.index}`,
      local_image: `local-${body.index}-${body.client_id}`, global_image: `global-${body.index}`,
    }) }));
  }));
  await h.flush();
  h.choose(3); h.choose(4); await h.flush();
  assert.equal(h.requests.length, 2);
  assert.equal(h.requests[0].signal.aborted, true);
  pending[1](); await h.settle();
  pending[0](); await h.settle();
  assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('原始')).props.src, 'clean-3');
  assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('局部')).props.src, 'local-3-0');
  assert.equal(h.find(n => n.type === 'img' && n.props.alt.includes('完整')).props.src, 'global-3');
  h.choose(5); await h.flush(); h.unmount();
  assert.equal(h.requests.at(-1).signal.aborted, true);
});

test('旧后端只返回两图时提示重启，不把完整触发器冒充训练投毒图', async () => {
  const h = harness({}, async () => ({ json: async () => ({
    total: 100, clean_image: 'clean', poisoned_image: 'full-only',
  }) }));
  await h.flush();
  assert.ok(h.find(n => n.type === 'Alert' && n.props.title.includes('重启后端')));
  assert.equal(h.find(n => n.type === 'img'), undefined);
});

test('预览区域可收起展开，收起时停止请求，展开保留图片编号', async () => {
  const h = harness(); await h.flush(); h.choose(8); await h.flush();
  h.click('收起');
  assert.equal(h.find(n => n.type === 'Card').props.styles.body.display, 'none');
  const count = h.requests.length;
  h.edit({ seed: 99 }); await h.flush();
  assert.equal(h.requests.length, count);
  h.click('展开'); await h.flush();
  assert.equal(h.requests.at(-1).body.index, 7);
  assert.notEqual(h.find(n => n.type === 'Card').props.styles.body.display, 'none');
});

test('预览错误内联显示中文，不残留旧图片，可手动重试', async () => {
  const h = harness({}, async () => { throw new Error('本地未找到完整数据集'); });
  await h.flush();
  assert.ok(h.find(n => n.type === 'Alert' && n.props.title === '本地未找到完整数据集'));
  assert.equal(h.find(n => n.type === 'img'), undefined);
  h.click('刷新预览'); await h.flush();
  assert.equal(h.requests.length, 2);
});

test('旧后端缺少预览接口时明确提示重启后端，而不是刷新列表', async () => {
  for (const message of ['Not Found', '请求失败（HTTP 404）']) {
    const h = harness({}, async () => { throw new errorHelpers.HttpRequestError(404, message); });
    await h.flush();
    assert.ok(h.find(n => n.type === 'Alert' && n.props.title.includes('重启后端')));
    assert.equal(h.find(n => n.type === 'img'), undefined);
  }
});

test('预览接口存在但数据集缺失时保留具体中文提示，不误报接口未加载', async () => {
  const message = '本地未找到完整数据集，请先准备或下载所选数据集，再刷新预览。';
  const h = harness({}, async () => { throw new errorHelpers.HttpRequestError(404, message); });
  await h.flush();
  assert.ok(h.find(n => n.type === 'Alert' && n.props.title === message));
  assert.equal(h.find(n => n.type === 'Alert' && n.props.title.includes('重启后端')), undefined);
});