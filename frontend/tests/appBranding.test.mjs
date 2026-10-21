import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const code = ts.transpileModule(readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

function harness({ home = false, role = 'user', signedIn = true } = {}) {
  const slots = ['config', home ? 'home' : 'workspace', signedIn ? { uid: 'test-uid', username: 'test', role } : null, false, false, true, 0];
  let cursor = 0;
  const jsx = (type, props, key) => ({ type, props, key });
  const exports = {};
  const imports = {
    react: { useState: () => { const index = cursor++; return [slots[index], value => { slots[index] = value; }]; }, useEffect() {} },
    'react/jsx-runtime': { jsx, jsxs: jsx },
    antd: { Layout: { Header: 'Header', Content: 'Content', Sider: 'Sider' }, Menu: 'Menu', Button: 'Button' },
    '@ant-design/icons': Object.fromEntries([
      'HistoryOutlined', 'SettingOutlined', 'AppstoreOutlined', 'FileTextOutlined', 'HomeOutlined', 'TeamOutlined',
    ].map(name => [name, name])),
    './App.css': {},
    './components/UserMenu': 'UserMenu',
    './api/auth': {}, './api/session': { setCsrfToken() {} },
    ...Object.fromEntries(['ConfigPage', 'HistoryPage', 'BatchPage', 'SummaryPage', 'LandingPage', 'LoginPage', 'AdminPage'].map(name => [
      `./pages/${name}`, name,
    ])),
  };
  runInNewContext(code, {
    exports,
    window: { location: { hash: home ? '#/' : '#/workspace' } },
    require(name) {
      assert.ok(Object.hasOwn(imports, name), `Unexpected import ${name}`);
      return imports[name];
    },
  });
  return {
    nodes() {
      cursor = 0;
      const nodes = [];
      function visit(node) {
        if (!node || typeof node !== 'object') return;
        if (Array.isArray(node)) { node.forEach(visit); return; }
        nodes.push(node);
        visit(node.props?.children);
      }
      visit(exports.default());
      return nodes;
    },
  };
}

function text(node) {
  if (typeof node === 'string') return node;
  if (Array.isArray(node)) return node.map(text).join('');
  return text(node?.props?.children ?? '');
}

test('头部显示联安智盾及完整平台名称，并使用统一的品牌图标', () => {
  const nodes = harness().nodes();
  assert.ok(nodes.find(node => node.type === 'Header' && node.props.className === 'platform-header'));
  assert.equal(text(nodes.find(node => node.props?.className === 'platform-brand__name')), '联安智盾');
  assert.equal(text(nodes.find(node => node.props?.className === 'platform-brand__description')),
    '联邦学习多层协同安全防护与可视化验证平台');
  const mark = nodes.find(node => node.type === 'img');
  assert.equal(mark.props.src, '/favicon.svg');
  assert.equal(mark.props.alt, '');
  assert.equal(mark.props.width, mark.props.height);
  assert.ok(!nodes.some(node => node.props?.href?.includes('github.com')));
  assert.ok(nodes.find(node => node.type === 'UserMenu'));
  assert.equal(text(nodes.find(node => node.props?.className === 'platform-home')), '返回首页');
});

test('品牌区调整不影响四个业务页面的菜单切换', () => {
  const app = harness();
  assert.ok(app.nodes().find(node => node.type === 'ConfigPage'));
  const menu = app.nodes().find(node => node.type === 'Menu');
  assert.equal(menu.props.items.length, 4);
  for (const [key, page] of [
    ['history', 'HistoryPage'], ['batch', 'BatchPage'], ['summary', 'SummaryPage'], ['config', 'ConfigPage'],
  ]) {
    menu.props.onClick({ key });
    const nodes = app.nodes();
    assert.ok(nodes.find(node => node.type === page));
    assert.equal(nodes.find(node => node.type === 'Menu').props.selectedKeys[0], key);
    assert.equal(nodes.find(node => node.type === 'main').key, key);
  }
});

test('工作区使用独立滚动容器，不再用固定高度白框包裹长表单', () => {
  const nodes = harness().nodes();
  assert.ok(nodes.find(node => node.type === 'aside' && node.props['aria-label'] === '主导航'));
  const main = nodes.find(node => node.type === 'main');
  assert.equal(main.props.className, 'platform-main');
  assert.equal(main.props.id, 'main-content');
  const content = nodes.find(node => node.type === 'Content');
  assert.equal(content.props.className, 'platform-content');
  assert.equal(content.props.style, undefined);
  assert.ok(!nodes.some(node => node.type === 'style'));
});

test('默认首页与登录入口不挂载未认证工作台', () => {
  const nodes = harness({ home: true, signedIn: false }).nodes();
  const landing = nodes.find(node => node.type === 'LandingPage');
  assert.ok(landing);
  assert.equal(landing.props.authenticated, false);
  assert.ok(!nodes.some(node => node.type === 'ConfigPage'));
});

test('用户与任务总览仅向管理员显示', () => {
  const nodes = harness({ role: 'admin' }).nodes();
  const menu = nodes.find(node => node.type === 'Menu');
  assert.equal(menu.props.items.length, 5);
  assert.equal(menu.props.items.at(-1).label, '用户与任务总览');
});

test('浏览器标题、描述及 favicon 与头部品牌保持一致', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const name = '联安智盾 联邦学习多层协同安全防护与可视化验证平台';
  assert.ok(html.includes(`<title>${name}</title>`));
  assert.ok(html.includes(`name="description" content="${name}"`));
  assert.match(html, /<link rel="icon" type="image\/svg\+xml" href="\/favicon\.svg"/);
  const favicon = readFileSync(new URL('../public/favicon.svg', import.meta.url), 'utf8');
  assert.match(favicon, /viewBox="0 0 64 64"/);
  assert.doesNotMatch(html, /联邦学习后门攻击防御系统|FL Backdoor Attack/);
});