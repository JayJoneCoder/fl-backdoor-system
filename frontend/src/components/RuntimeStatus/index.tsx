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
          <Alert showIcon type={!report.cpu_available ? 'error' : report.gpu_available ? 'success' : 'warning'} title={!report.cpu_available ? '运行环境不可用' : report.gpu_available ? 'GPU 实际运算自检通过' : 'GPU 不可用，自动模式将在启动前选择 CPU'} description={[...report.warnings, ...report.guidance].join('；')} />
          <Typography.Text>CUDA Toolkit 不是 GPU 可用性的依据。更换 PyTorch 请先关闭平台，再通过启动器 --setup 配置；GPU 训练中出现故障不会静默回退。</Typography.Text>
          <Typography.Text>当前已保存设备配置：{mode}；启动器策略：{report.selected_device}。请在单个实验配置的「实验管理 → 训练计算设备」中修改并保存，或在批量 JSON 中设置 runtime-device；此窗口只读，不覆盖未保存的表单。</Typography.Text>
        </>}
        <Button loading={loading} onClick={() => void refresh()}>重新检测</Button>
      </Space>
    </Modal>
  </>;
}