import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const exports = {};
runInNewContext(ts.transpileModule(readFileSync(new URL('../src/theme.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023 },
}).outputText, { exports });

test('平台主题为表单、表格和导航提供统一的视觉参数', () => {
  const { token, components } = exports.platformTheme;
  assert.equal(token.colorPrimary, token.colorInfo);
  assert.equal(token.colorBgContainer, '#ffffff');
  assert.ok(token.controlHeight >= 36);
  assert.ok(token.borderRadius <= 8);
  assert.equal(components.Button.primaryShadow, 'none');
  for (const component of ['Card', 'Form', 'Menu', 'Table', 'Tag', 'Statistic']) {
    assert.ok(components[component], `${component} shares the platform theme`);
  }
});

test('应用入口启用中文组件文本及统一主题', () => {
  const main = readFileSync(new URL('../src/main.tsx', import.meta.url), 'utf8');
  assert.match(main, /antd\/locale\/zh_CN/);
  assert.match(main, /<ConfigProvider locale=\{zhCN\} theme=\{platformTheme\}>/);
});