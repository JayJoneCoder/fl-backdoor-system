export class HttpRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'HttpRequestError';
    this.status = status;
  }
}

// fetch rejects network failures, not HTTP errors. Preserve explicit conflict
// handling while requiring every other response to be successful.
export const checkedFetch = async (
  input: RequestInfo | URL,
  init?: RequestInit,
  allowedStatuses: number[] = [],
): Promise<Response> => {
  const response = await fetch(input, init);
  if (!response.ok && !allowedStatuses.includes(response.status)) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const data: unknown = await response.json();
      if (data && typeof data === 'object' && 'detail' in data) {
        const detail = (data as { detail: unknown }).detail;
        if (typeof detail === 'string') message = detail;
      }
    } catch {
      // A non-JSON error response is still an HTTP failure.
    }
    throw new HttpRequestError(response.status, message);
  }
  return response;
};

// Translate only at the presentation boundary. Keep HTTP errors and run logs
// unchanged so their original diagnostics remain available for debugging.
const ERROR_TEXT: Record<string, string> = {
  'Configuration content cannot be empty.': '配置内容不能为空。',
  'Missing project.name.': '配置缺少项目名称（project.name）。',
  'Flower config and components must be TOML tables.': 'Flower 的配置和组件必须使用 TOML 表结构。',
  'Configuration must include project and tool.flwr.app config/components tables.': '配置必须包含 project 和 tool.flwr.app 下的 config、components 表。',
  'Configuration must be an object.': '配置必须为键值对象。',
  'Configuration updates must be an object.': '配置更新必须为键值对象。',
  'Invalid batch experiment JSON format': '批量实验 JSON 格式不正确。',
  'Backup name already exists; choose another name.': '备份名称已存在，请使用其他名称。',
  'The active project configuration cannot be used as a backup.': '不能将当前项目主配置作为备份操作。',
  'Backup filename must end in .toml.': '备份文件名必须以 .toml 结尾。',
  'Backup name must be text.': '备份名称必须是文本。',
  'Invalid automatic backup path.': '自动备份路径不合法。',
  'on_conflict must be ask, auto or overwrite.': '请选择询问、自动重命名或覆盖的备份冲突处理方式。',
  'An experiment is running or cleanup is not finished': '已有实验正在运行或尚未完成清理，请稍后重试。',
  'Experiment already running': '已有实验正在运行，请先停止当前实验。',
  'A batch task is running or stopping': '批量任务正在运行或清理中，请等待完成后再启动单个实验。',
  'A batch task is already running or stopping': '已有批量任务正在运行或清理中，请稍后重试。',
  'Experiment name is required': '请输入实验名称。',
  'Experiment config must be an object': '实验配置必须为键值对象。',
  'Experiment already exists. Choose a new name.': '实验名称已存在，请使用新名称；旧结果未被修改。',
  'Experiment not found or no valid data': '找不到该实验，或实验尚未生成有效数据。',
  'CSV file not found yet': '实验尚未生成 CSV，请稍后重试。',
  'Only .toml files are accepted': '仅支持上传 .toml 配置文件。',
  "Missing 'content' field": '缺少配置内容，请先填写或上传配置。',
  'Missing content': '配置内容不能为空。',
  'Missing filename': '请选择备份文件。',
  "Missing 'from' filename": '请选择需要重命名的备份文件。',
  'Backup not found': '找不到该备份，请刷新备份列表。',
  'Batch task not found': '找不到该批量任务。',
  'Experiment not found': '找不到该实验。',
  'Summary not found': '找不到该总结。',
  'Summary already exists; choose a new name': '总结名称已存在，请使用新名称。',
  'Invalid summary data': '总结数据无效，请检查所选实验是否生成了有效结果。',
  'No images found in this summary': '该总结暂无图表，请先生成图表。',
  'No images found in this experiment': '该实验暂无图表，请先生成图表。',
  'No files found in this experiment': '该实验目录中没有可下载的文件。',
  'No experiments selected': '请至少选择一个实验。',
  'Experiments must be a list of names': '实验列表必须由实验名称组成。',
  'No experiments provided': '请提供至少一个实验。',
  'Backend restarted; task was interrupted and was not automatically resumed.': '后端已重启，该任务已中断，未自动恢复运行。',
  'Not started before backend restart.': '后端重启前未启动此实验。',
  'Execution details were not recorded by this older version.': '旧版本未记录逐项执行明细，无法确认该实验的执行状态。',
  'Batch cancelled before completion.': '批量任务已取消，此实验未完成。',
  'Owned cleanup is not finished. Retry cancellation.': '实验进程尚未清理完成，请重试取消／清理。',
  'Owned processes are still exiting; cleanup must be retried.': '实验进程仍在退出，请重试清理。',
  'Owned process termination is not complete.': '实验进程尚未完全退出，请重试清理。',
  'Some owned child processes could not be stopped.': '部分实验子进程未能停止，请重试清理。',
  'Invalid experiment run metadata.': '实验运行记录格式不正确。',
  'Invalid experiment run ID.': '实验 UUID 不合法。',
  'Experiment CSV files already exist; refusing to overwrite results.': '实验 CSV 已存在，已阻止覆盖旧结果，请使用新实验名。',
  'Sub-pattern count must be a square fitting trigger-size, or supply a sub-pattern size.': '子块数量必须是与触发器尺寸匹配的平方数，或手动指定子块尺寸。',
  'Sub-pattern size must fit trigger-size.': '子块尺寸不能超过触发器尺寸。',
  'Sub-pattern count and size do not match trigger-size.': '子块数量、子块尺寸与触发器尺寸不匹配。',
  'Krum configuration requires num-clients > 2*f+2 and 1 <= k <= num-clients-f-2.': 'Krum 配置不合法：客户端数 n 必须大于 2f+2，且选择数 k 必须满足 1 ≤ k ≤ n−f−2（f 为预估恶意数）。',
};

const NAME_LABELS: Record<string, string> = {
  Name: '名称', 'Experiment name': '实验名称', 'Summary name': '总结名称',
  'Backup filename': '备份文件名', 'Backup name': '备份名称', 'Automatic backup': '自动备份',
};

const FIELD_LABELS: Record<string, string> = {
  'batch-size': '批次大小', 'num-clients': '客户端总数', 'num-server-rounds': '训练轮数',
  'local-epochs': '本地训练轮数', 'learning-rate': '学习率', seed: '随机种子',
  'target-label': '目标标签', 'poison-rate': '投毒比例', 'malicious-count': '恶意客户端数',
  'malicious-ratio': '恶意客户端比例', 'attack-fixed-clients': '固定恶意客户端编号',
  'trigger-size': '触发器尺寸', 'defense-num-malicious': '预估恶意数',
  'defense-krum-k': 'Krum 选择数', 'defense-trim-k': '修剪数量', 'defense-trim-ratio': '修剪比例',
};

const translateMessage = (raw: string): string | null => {
  const text = raw.trim();
  if (Object.hasOwn(ERROR_TEXT, text)) return ERROR_TEXT[text];
  const conflict = text.match(/^Experiment "(.*)" already exists(?:\. Choose a new name; old results were not changed\.| and cannot be overwritten\.)$/);
  if (conflict) return `实验“${conflict[1]}”已存在，请使用新名称；旧结果未被修改。`;
  const worker = text.match(/^Experiment worker exited with code (.+)\. See run\.log for details\.$/);
  if (worker) return `实验运行失败（退出码 ${worker[1]}），请查看运行日志 run.log。`;
  const krum = text.match(/^Krum requires n > 2\*f \+ 2 after detection, got n=(\d+), f=(\d+)\.$/);
  if (krum) return `检测后仅剩 ${krum[1]} 个客户端更新，不满足 Krum 的 n > 2f+2 条件（预估恶意数 f=${krum[2]}）。`;
  const backup = text.match(/^Backup file ['"](.*)['"] not found$/);
  if (backup) return `找不到备份文件“${backup[1]}”。`;
  const missingExperiment = text.match(/^Experiment (.+) not found$/);
  if (missingExperiment) return `找不到实验“${missingExperiment[1]}”。`;
  const duplicate = text.match(/^Duplicate experiment name: (.+)$/);
  if (duplicate) return `批量配置中存在重复的实验名称：${duplicate[1]}。`;
  const indexed = text.match(/^Experiment (\d+): (.+)$/);
  if (indexed) return `第 ${indexed[1]} 个实验：${translateMessage(indexed[2]) ?? '配置不合法，请检查名称和参数。'}`;
  if (text === 'name is required') return '实验名称不能为空。';
  const failure = text.match(/^Experiment (.+?) failed(?:: (.*)|\.)$/);
  if (failure) return `实验“${failure[1]}”失败。${failure[2] ? translateMessage(failure[2]) ?? '请查看运行日志。' : '请查看运行日志。'}`;

  const wrappers: [string, string][] = [
    ['Cleanup failed: ', '清理失败：'], ['Experiment monitor failed: ', '实验状态监控失败：'],
    ['Batch controller failed: ', '批量调度失败：'], ['Internal server error: ', '后端处理失败：'],
    ['Plot generation failed: ', '图表生成失败：'], ['Summary generation failed: ', '总结生成失败：'],
    ['Summary failed: ', '总结生成失败：'],
  ];
  for (const [prefix, translated] of wrappers) {
    if (text.startsWith(prefix)) return translated + (translateMessage(text.slice(prefix.length)) ?? '请查看运行日志，并在确认问题后重试。');
  }
  if (/^(?:Invalid TOML(?:$|:)|bad TOML)/i.test(text)) return 'TOML 配置格式不正确，请检查语法、字段类型及行列位置。';
  if (/^Missing tool\.flwr\.app\.components\./.test(text)) return '配置缺少 Flower 的服务端或客户端组件入口。';

  const name = text.match(/^(Name|Experiment name|Summary name|Backup filename|Backup name|Automatic backup) (.+)$/);
  if (name) {
    const label = NAME_LABELS[name[1]];
    const reason = name[2];
    const length = reason.match(/^must contain 1 to (\d+) characters\.$/);
    if (length) return `${label}长度必须为 1～${length[1]} 个字符。`;
    if (reason.includes('surrounding whitespace')) return `${label}不能以点或短横线开头，也不能带首尾空格。`;
    if (reason.includes('must be a filename')) return `${label}不能包含路径分隔符、控制字符或系统保留字符。`;
    if (reason.includes('Windows device name')) return `${label}不能使用 Windows 保留设备名。`;
    if (reason.includes('reserved for application data')) return `${label}已被系统保留，请使用其他名称。`;
    if (reason.includes('links are not allowed') || reason.includes('resolved safely')) return `${label}路径不安全，不能越出指定目录或使用链接。`;
  }
  const field = text.match(/^([a-z][a-z0-9-]*): (.+)$/);
  if (field) {
    const [, key, reason] = field;
    const minimum = reason.match(/^must be at least (\d+)\.$/);
    const range = reason.match(/^must be in (\[[^\]]+[)\]])\.$/);
    const below = reason.match(/^must be below (\d+) for (.+)\.$/);
    let translated: string;
    if (minimum) translated = `不能小于 ${minimum[1]}。`;
    else if (range) translated = `必须在 ${range[1]} 范围内。`;
    else if (below) translated = `在 ${below[2]} 数据集中必须小于 ${below[1]}。`;
    else if (reason.includes('true or false')) translated = '必须填写 true 或 false（布尔值）。';
    else if (reason.includes('positive integer')) translated = '必须为正整数。';
    else if (/whole number|TOML integer|list must contain integers/.test(reason)) translated = '必须填写整数，不能使用小数、布尔值或字符串代替。';
    else if (/finite|expected a number|TOML number/.test(reason)) translated = '必须填写有限数值，不能使用空值、无穷大或非数值。';
    else if (reason === 'must be non-negative.') translated = '不能小于 0。';
    else if (reason === 'must be positive.') translated = '必须大于 0。';
    else if (reason.includes('empty')) translated = '不能为空；可选数字留空时请省略该配置项。';
    else if (reason.startsWith('unsupported value')) translated = '所选值不受支持，请从可用选项中选择。';
    else if (reason === 'expected text.') translated = '必须填写文本。';
    else if (/cannot exceed num-clients|IDs must be within num-clients/.test(reason)) translated = '不能超过客户端总数，客户端编号应在有效范围内。';
    else if (reason.includes('coordinates')) translated = '必须填写两个非负整数坐标。';
    else if (reason.includes('comma-separated')) translated = '请使用逗号分隔的非负整数客户端编号。';
    else if (reason.includes('does not fit the selected dataset')) translated = '触发器尺寸超出了所选数据集的图像范围。';
    else if (reason.includes('2 * trim_k')) translated = '两倍修剪数量必须小于客户端总数。';
    else translated = '参数不合法，请检查类型、范围及与其他参数的组合。';
    const label = Object.hasOwn(FIELD_LABELS, key) ? `${FIELD_LABELS[key]}（${key}）` : key;
    return `${label}：${translated}`;
  }
  // Existing application messages are already localized. Never append unknown
  // English exception text to a Chinese toast or alert.
  if (/^[\u3400-\u9fff]/.test(text) || /^请求失败（HTTP \d+）$/.test(text)) return text;
  return null;
};

export const getUserErrorMessage = (error: unknown, fallback = '操作失败，请查看运行日志或稍后重试。'): string => {
  if (typeof error === 'string') return translateMessage(error) ?? fallback;
  if (!error || typeof error !== 'object') return fallback;
  const value = error as {
    message?: unknown; code?: unknown; status?: number;
    response?: { status?: number; data?: { detail?: unknown } };
  };
  const detail = value.response?.data?.detail;
  if (typeof detail === 'string') {
    const translated = translateMessage(detail);
    if (translated) return translated;
  }
  if (Array.isArray(detail)) {
    // FastAPI/Pydantic validation errors: keep field paths, not English internals.
    const fields = detail.flatMap(item => {
      if (!item || typeof item !== 'object' || !Array.isArray(item.loc)) return [];
      return [item.loc.filter((part: unknown) => part !== 'body' && part !== 'query').join('.')];
    }).filter(Boolean);
    return fields.length ? `请求参数不合法，请检查：${[...new Set(fields)].join('、')}。` : '请求参数不合法，请检查必填项、类型和范围。';
  }
  const message = typeof value.message === 'string' ? value.message : '';
  const translated = translateMessage(message);
  if (translated) return translated;
  if (value.code === 'ECONNABORTED' || value.code === 'ETIMEDOUT' || /timeout|timed out/i.test(message)) return '请求超时，请检查后端状态后重试。';
  if (value.code === 'ERR_NETWORK' || /NetworkError|Network Error|Failed to fetch|Load failed|network unavailable/i.test(message)) return '无法连接后端，请确认后端服务已启动，并检查网络连接。';
  const status = value.response?.status ?? value.status;
  if (status === 404) return '请求的资源不存在，请刷新列表后重试。';
  if (status === 409) return '名称已存在或当前有任务占用，请更换名称或等待清理完成。';
  if (status === 422 || status === 400) return '请求参数不合法，请检查名称和配置。';
  if (status && status >= 500) return `后端处理失败（HTTP ${status}），请查看运行日志。`;
  return fallback;
};

export const getConfigErrorMessage = (error: unknown): string => getUserErrorMessage(error, '配置操作失败，请检查配置或查看后端日志。');