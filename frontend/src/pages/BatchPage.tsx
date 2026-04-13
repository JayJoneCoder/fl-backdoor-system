import React, { useState } from 'react';
import { Card, Upload, Button, Space, message, Table, Progress, Alert } from 'antd';
import { UploadOutlined, PlayCircleOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { parseBatchConfig, runBatchExperiments } from '../api/client';

const BatchPage: React.FC = () => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleUpload = async () => {
    if (fileList.length === 0) return;
    const file = fileList[0].originFileObj;
    if (!file) return;
    setLoading(true);
    try {
      const res = await parseBatchConfig(file);
      setExperiments(res.data.experiments);
      message.success(`成功解析 ${res.data.count} 个实验`);
    } catch (error) {
      message.error('解析失败，请检查 JSON 格式');
    } finally {
      setLoading(false);
    }
  };

  const handleRun = async () => {
    if (experiments.length === 0) return;
    setRunning(true);
    setProgress(0);
    try {
      await runBatchExperiments(experiments);
      message.success('批量实验已开始');
      // 模拟进度（后端目前没有提供进度接口，这里简单模拟）
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setRunning(false);
            return 100;
          }
          return prev + 10;
        });
      }, 1000);
    } catch (error) {
      message.error('启动失败');
      setRunning(false);
    }
  };

  const columns = [
    { title: '实验名称', dataIndex: 'name', key: 'name' },
    {
      title: '配置',
      dataIndex: 'config',
      key: 'config',
      render: (config: any) => (
        <pre style={{ maxHeight: 100, overflow: 'auto', fontSize: 11 }}>
          {JSON.stringify(config, null, 2)}
        </pre>
      ),
    },
  ];

  return (
    <div>
      <h2>批量实验</h2>
      <Card>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            message="上传 JSON 配置文件"
            description='文件格式示例：{ "experiments": [ { "name": "exp1", "attack": "badnets" } ], "common": { "dataset": "mnist" } }'
            type="info"
            showIcon
          />
          <Space>
            <Upload
              fileList={fileList}
              beforeUpload={() => false}
              onChange={({ fileList }) => setFileList(fileList)}
              maxCount={1}
            >
              <Button icon={<UploadOutlined />}>选择 JSON 文件</Button>
            </Upload>
            <Button type="primary" onClick={handleUpload} loading={loading}>
              解析
            </Button>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleRun}
              disabled={experiments.length === 0}
              loading={running}
              danger
            >
              运行批量实验
            </Button>
          </Space>

          {experiments.length > 0 && (
            <>
              <h3>实验列表 ({experiments.length})</h3>
              <Table
                dataSource={experiments}
                columns={columns}
                rowKey="name"
                pagination={false}
                size="small"
              />
            </>
          )}

          {running && (
            <Progress percent={progress} status="active" style={{ marginTop: 16 }} />
          )}
        </Space>
      </Card>
    </div>
  );
};

export default BatchPage;