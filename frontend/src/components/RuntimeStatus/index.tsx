import { useEffect, useState } from 'react';
import { Alert, Button, Modal, Space, Typography } from 'antd';
import { apiClient, getConfig } from '../../api/client';
import { getUserErrorMessage } from '../../api/checkedFetch';

interface Report {
  torch: string | null;
  torchvision: string | null;
  cuda_build: string | null;
  cpu_available: boolean;
  gpu_available: boolean;
  selected_device: string;
  warnings: string[];
  guidance: string[];
  simulation_backend?: 'local' | 'ray';
  ray?: {
    checked: boolean;
    ok: boolean | null;
    error: string;
    checked_at_text: string;
    ray: string | null;
    seconds: number | null;
    ray_temp_dir: string;
  };
}

export default function RuntimeStatus() {
  const [report, setReport] = useState<Report | null>(null);
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState('auto');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const refresh = async () => {
    setLoading(true);
    try {
      const [runtime, config] = await Promise.all([apiClient.get<Report>('/api/runtime'), getConfig()]);
      setReport(runtime.data);
      setMode(config.data['runtime-device'] || 'auto');
      setError('');
      if (!runtime.data.gpu_available || !runtime.data.cpu_available) setOpen(true);
    } catch (err) { setError(getUserErrorMessage(err)); setOpen(true); }
    finally { setLoading(false); }
  };
  useEffect(() => { void refresh(); }, []);
  return <>
    <Button type="text" onClick={() => { setOpen(true); if (!report) void refresh(); }}>运行环境{report ? ` · ${report.gpu_available ? 'GPU 就绪' : 'CPU'}` : ''}</Button>
    <Modal title="计算设备与运行环境" open={open} onCancel={() => setOpen(false)} footer={<Button onClick={() => setOpen(false)}>已了解</Button>}>
      <Space orientation="vertical" style={{ width: '100%' }}>
        {error && <Alert type="error" showIcon title={error} />}
        {report && <>
          <Typography.Text>PyTorch {report.torch || '未安装'} / torchvision {report.torchvision || '未安装'} / CUDA 构建 {report.cuda_build || '无'}</Typography.Text>
          <Typography.Text>
            Flower 仿真后端：{report.simulation_backend === 'ray'
              ? `Ray ${report.ray?.ray || ''} · ${report.ray?.checked
                ? (report.ray.ok ? `自检通过（${report.ray.seconds ?? '?'} 秒）` : '自检失败')
                : '尚未自检（首次实验前自动检测，约 10–20 秒）'}`
              : '本地执行（不依赖 Ray；客户端顺序运行，结果可复现）'}
            {report.simulation_backend === 'ray' && report.ray?.checked_at_text
              ? ` · ${report.ray.checked_at_text}` : ''}
          </Typography.Text>
          {report.simulation_backend === 'ray' && report.ray?.checked && report.ray.ok === false && (
            <Alert type="error" showIcon title="Ray 自检失败（Windows 下为实验性支持）"
              description={`${report.ray.error} 排查：把安装目录加入杀毒软件白名单；确认 VC++ 2015-2022 运行库已安装；Ray 临时目录已切换到 ${report.ray.ray_temp_dir || 'ASCII 路径'}。仍失败可在 WSL2 中运行或换一台机器。`} />
          )}
          <Alert showIcon type={!report.cpu_available ? 'error' : report.gpu_available ? 'success' : 'warning'} title={!report.cpu_available ? '运行环境不可用' : report.gpu_available ? 'GPU 实际运算自检通过' : 'GPU 不可用，自动模式将在启动前选择 CPU'} description={[...report.warnings, ...report.guidance].join('；')} />
          <Typography.Text>CUDA Toolkit 不是 GPU 可用性的依据。更换 PyTorch 请先关闭平台，再通过启动器 --setup 配置；GPU 训练中出现故障不会静默回退。</Typography.Text>
          <Typography.Text>当前已保存设备配置：{mode}；启动器策略：{report.selected_device}。请在单个实验配置的「实验管理 → 训练计算设备」中修改并保存，或在批量 JSON 中设置 runtime-device；此窗口只读，不覆盖未保存的表单。</Typography.Text>
        </>}
        <Button loading={loading} onClick={() => void refresh()}>重新检测</Button>
      </Space>
    </Modal>
  </>;
}