import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/api/checkedFetch.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 },
}).outputText;
const { checkedFetch, HttpRequestError, getConfigErrorMessage, getUserErrorMessage } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`
);

test('成功响应保持原样，并传递请求参数', async t => {
  const response = new Response(JSON.stringify({ status: 'success' }), { status: 200 });
  let received;
  t.mock.method(globalThis, 'fetch', async (...args) => { received = args; return response; });
  const options = { method: 'POST', body: 'content' };
  assert.equal(await checkedFetch('/api/config/raw', options), response);
  assert.deepEqual(received, ['/api/config/raw', options]);
  assert.deepEqual(await response.json(), { status: 'success' });
});

test('HTTP 400 不会当成成功，返回后端校验信息', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ detail: 'batch-size: invalid' }), { status: 400 }));
  await assert.rejects(checkedFetch('/api/config/raw'), error => {
    assert.ok(error instanceof HttpRequestError);
    assert.equal(error.status, 400);
    assert.equal(error.message, 'batch-size: invalid');
    return true;
  });
});

test('HTTP 500 非 JSON 内容仍然失败', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('Internal server error', { status: 500 }));
  await assert.rejects(checkedFetch('/api/config/raw'), /HTTP 500/);
});

test('仅显式允许的 409 留给重名冲突处理', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('{}', { status: 409 }));
  assert.equal((await checkedFetch('/api/config/backups', {}, [409])).status, 409);
  await assert.rejects(checkedFetch('/api/config/backups'), error => error.status === 409);
});

test('网络异常不会被吞掉', async t => {
  const error = new Error('network unavailable');
  t.mock.method(globalThis, 'fetch', async () => { throw error; });
  await assert.rejects(checkedFetch('/api/config'), received => received === error);
});

test('配置错误优先翻译后端 detail，支持 axios 与 fetch 异常', () => {
  assert.match(getConfigErrorMessage({ response: { data: { detail: 'batch-size: must be a positive integer.' } } }), /批次大小.*必须为正整数/);
  assert.match(getConfigErrorMessage(new HttpRequestError(400, 'bad TOML')), /TOML 配置格式不正确/);
  assert.equal(getConfigErrorMessage(null), '配置操作失败，请检查配置或查看后端日志。');
});

test('同名实验、非法路径与备份错误给出中文原因，不修改原异常', () => {
  const raw = 'Experiment "实验一" already exists. Choose a new name; old results were not changed.';
  const error = { response: { status: 409, data: { detail: raw } } };
  assert.equal(getUserErrorMessage(error), '实验“实验一”已存在，请使用新名称；旧结果未被修改。');
  assert.equal(error.response.data.detail, raw);
  assert.match(getUserErrorMessage('Experiment name must be a filename, not a path, and contain no reserved characters.'), /实验名称不能包含路径/);
  assert.match(getUserErrorMessage('Summary name must stay inside its data directory; links are not allowed.'), /总结名称路径不安全/);
  assert.match(getUserErrorMessage('The active project configuration cannot be used as a backup.'), /不能将当前项目主配置作为备份/);
  assert.equal(getUserErrorMessage("Backup file '测试.toml' not found"), '找不到备份文件“测试.toml”。');
});

test('配置类型、范围、布尔值、数据集及 Krum 约束可读且为中文', () => {
  const cases = [
    ['batch-size: expected a whole number.', /批次大小.*整数/],
    ['poison-rate: must be in [0, 1].', /必须在 \[0, 1\] 范围内/],
    ['fraction-evaluate: must be in [0, 1).', /必须在 \[0, 1\) 范围内/],
    ['enabled: expected true or false, not 1.', /必须填写 true 或 false/],
    ['learning-rate: number must be finite.', /学习率.*有限数值/],
    ['target-label: must be below 10 for mnist.', /目标标签.*必须小于 10/],
    ['fcba-num-sub-blocks: must be at least 2.', /不能小于 2/],
    ['attack: unsupported value unknown; expected one of [].', /所选值不受支持/],
    ['Krum configuration requires num-clients > 2*f+2 and 1 <= k <= num-clients-f-2.', /Krum 配置不合法.*2f\+2/],
    ['Krum requires n > 2*f + 2 after detection, got n=7, f=3.', /仅剩 7 个.*f=3/],
  ];
  for (const [raw, expected] of cases) assert.match(getUserErrorMessage(raw), expected, raw);
});

test('批量、清理失败及重启中断会翻译，未知内部错误使用中文兜底', () => {
  assert.match(getUserErrorMessage('Cleanup failed: Owned process termination is not complete.'), /清理失败：实验进程尚未完全退出/);
  assert.match(getUserErrorMessage('Experiment demo failed: Experiment worker exited with code 1. See run.log for details.'), /实验“demo”失败。实验运行失败（退出码 1）/);
  assert.match(getUserErrorMessage('Backend restarted; task was interrupted and was not automatically resumed.'), /后端已重启/);
  assert.equal(getUserErrorMessage(new Error('Traceback: private internal exception')), '操作失败，请查看运行日志或稍后重试。');
  assert.equal(getUserErrorMessage('未知配置项，请检查。'), '未知配置项，请检查。');
  assert.equal(getUserErrorMessage('__proto__'), '操作失败，请查看运行日志或稍后重试。');
});

test('网络与超时错误和未知 HTTP 错误均给出中文提示', () => {
  assert.match(getUserErrorMessage(new TypeError('Failed to fetch')), /无法连接后端/);
  assert.match(getUserErrorMessage({ code: 'ERR_NETWORK' }), /无法连接后端/);
  assert.match(getUserErrorMessage({ code: 'ECONNABORTED', message: 'timeout of 1000ms exceeded' }), /请求超时/);
  assert.match(getUserErrorMessage({ response: { status: 500, data: { detail: 'private exception' } } }), /后端处理失败（HTTP 500）/);
  assert.match(getUserErrorMessage({ response: { status: 409 } }), /名称已存在/);
});

test('结构化字段校验错误列出字段路径，不显示英文校验详情', () => {
  const error = { response: { status: 422, data: { detail: [
    { loc: ['body', 'name'], msg: 'Field required' },
    { loc: ['body', 'config', 'batch-size'], msg: 'Input should be an integer' },
  ] } } };
  assert.equal(getUserErrorMessage(error), '请求参数不合法，请检查：name、config.batch-size。');
});