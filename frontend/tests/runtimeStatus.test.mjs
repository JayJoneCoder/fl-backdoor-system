import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const source = readFileSync(new URL('../src/components/RuntimeStatus/index.tsx', import.meta.url), 'utf8');
const code = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

async function harness({ gpu = false, cpu = true, fail = false } = {}) {
  const slots = [], effects = [], saves = [];
  let cursor = 0;
  const jsx = (type, props) => ({ type, props });
  const imports = {
    react: {
      useState(initial) { const index = cursor++; if (!(index in slots)) slots[index] = initial; return [slots[index], value => { slots[index] = value; }]; },
      useEffect(effect) { const index = cursor++; if (!(index in slots)) { slots[index] = true; effects.push(effect); } },
    },
    'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' },
    antd: { Alert: 'Alert', Button: 'Button', Modal: 'Modal', Select: 'Select', Space: 'Space', Typography: { Text: 'Text' } },
    '../../api/client': {
      apiClient: { get: async () => { if (fail) throw new Error('offline'); return { data: { torch: '2.8.0', torchvision: '0.23.0', cpu_available: cpu, gpu_available: gpu, warnings: gpu ? [] : ['CPU wheel'], guidance: ['换 CUDA 版'], selected_device: 'auto' } }; } },
      getConfig: async () => ({ data: { 'runtime-device': 'cpu' } }),
      updateConfig: async value => { saves.push(value); },
    },
    '../../api/checkedFetch': { getUserErrorMessage: error => error.message },
  };
  const exports = {};
  runInNewContext(code, { exports, require: name => { assert.ok(name in imports, name); return imports[name]; } });
  function nodes() {
    cursor = 0;
    const result = [];
    const visit = node => {
      if (Array.isArray(node)) return node.forEach(visit);
      if (!node || typeof node !== 'object') return;
      result.push(node); visit(node.props?.children);
    };
    visit(exports.default());
    return result;
  }
  nodes(); effects.splice(0).forEach(effect => effect());
  await new Promise(resolve => setImmediate(resolve));
  return { nodes, saves };
}

test('CPU-only 环境自动弹出中文警告和解决指引，读取扁平设备配置', async () => {
  const app = await harness();
  const nodes = app.nodes();
  assert.equal(nodes.find(node => node.type === 'Modal').props.open, true);
  assert.equal(nodes.find(node => node.type === 'Alert').props.type, 'warning');
  assert.match(nodes.find(node => node.type === 'Alert').props.description, /CPU wheel.*换 CUDA 版/);
  const text = nodes.filter(node => node.type === 'Text').map(node => JSON.stringify(node.props.children)).join('');
  assert.match(text, /当前已保存设备配置.*cpu/);
});

test('GPU 自检通过不自动打断工作，诊断窗口只读不覆盖实验表单', async () => {
  const app = await harness({ gpu: true });
  assert.equal(app.nodes().find(node => node.type === 'Modal').props.open, false);
  assert.equal(app.saves.length, 0);
  assert.equal(app.nodes().some(node => node.type === 'Select'), false);
  assert.match(source, /runtime-device/);
  const form = readFileSync(new URL('../src/components/ConfigForm/index.tsx', import.meta.url), 'utf8');
  assert.match(form, /'runtime-device': '训练计算设备'/);
});

test('环境缺失显示错误，不把缺包当作可用 CPU', async () => {
  const app = await harness({ cpu: false });
  const alert = app.nodes().find(node => node.type === 'Alert');
  assert.equal(alert.props.type, 'error');
  assert.equal(alert.props.title, '运行环境不可用');
});

test('检测失败保留错误和手动重试入口', async () => {
  const app = await harness({ fail: true });
  assert.equal(app.nodes().find(node => node.type === 'Alert').props.title, 'offline');
  assert.ok(app.nodes().find(node => node.type === 'Button' && node.props.children === '重新检测'));
});