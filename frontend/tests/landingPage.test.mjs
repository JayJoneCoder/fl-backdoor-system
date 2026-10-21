import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

function render(file, imports = {}, props = {}) {
  const code = ts.transpileModule(readFileSync(new URL(file, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const jsx = (type, props) => typeof type === 'function' ? type(props) : { type, props };
  const exports = {};
  runInNewContext(code, { exports, require: name => {
    if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx };
    if (name === 'antd') return { Button: 'Button' };
    if (name === '@ant-design/icons' || name.endsWith('.css')) return {};
    assert.ok(name in imports); return imports[name];
  }, document: { getElementById: () => ({ scrollIntoView() {} }) } });
  return exports.default(props);
}

const text = node => typeof node === 'string' ? node : Array.isArray(node) ? node.map(text).join('') : text(node?.props?.children ?? '');
const nodes = tree => { const result = []; function visit(node) { if (!node || typeof node !== 'object') return; if (Array.isArray(node)) return node.forEach(visit); result.push(node); visit(node.props?.children); } visit(tree); return result; };

test('欢迎页显示完整作品名称、三层能力和真实登录入口', () => {
  let enters = 0;
  const tree = render('../src/pages/LandingPage.tsx', {}, { authenticated: false, onEnter: () => enters++ });
  const all = nodes(tree);
  assert.equal(text(all.find(n => n.type === 'h1')), '联安智盾');
  assert.ok(text(tree).includes('面向联邦学习的多层协同安全防护与可视化验证平台'));
  for (const label of ['客户端样本特征过滤', '服务端异常更新检测', '鲁棒聚合与模型保护', 'IID / Non-IID 数据划分', '典型安全场景验证', '实时训练监控', '批量实验与结果追溯']) assert.ok(text(tree).includes(label));
  const actions = all.filter(n => n.type === 'Button');
  assert.equal(actions.length, 3);
  actions.forEach(n => n.props.onClick());
  assert.equal(enters, 3);
  assert.ok(all.find(n => n.props?.id === 'core-capabilities'));
});

test('已登录用户可从欢迎页返回实验工作台', () => {
  const tree = render('../src/pages/LandingPage.tsx', {}, { authenticated: true, onEnter() {} });
  assert.ok(text(tree).includes('进入实验工作台'));
});

test('欢迎页与认证界面有桌面及移动布局，尊重减少动态效果设置', () => {
  const landing = readFileSync(new URL('../src/pages/LandingPage.css', import.meta.url), 'utf8');
  const login = readFileSync(new URL('../src/pages/LoginPage.css', import.meta.url), 'utf8');
  assert.match(landing, /max-width: 760px/);
  assert.match(landing, /max-width: 380px/);
  assert.match(landing, /min-width: 1650px/);
  assert.match(login, /max-width: 640px/);
  const global = readFileSync(new URL('../src/index.css', import.meta.url), 'utf8');
  assert.match(global, /prefers-reduced-motion: reduce/);
});