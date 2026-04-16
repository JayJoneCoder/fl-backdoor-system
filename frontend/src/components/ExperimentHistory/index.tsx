import React, { useEffect, useState } from 'react';
import {
  Card,
  Row,
  Col,
  Spin,
  message,
  Button,
  Modal,
  Table,
  Space,
  Divider,
  Image,
  Statistic,
  Descriptions,
  Tooltip,
  Dropdown,
} from 'antd';
import type { MenuProps } from 'antd';
import {
  BarChartOutlined,
  DownloadOutlined,
  FileTextOutlined,
  ExperimentOutlined,
  AimOutlined,
  FileImageOutlined,
  FileZipOutlined,
} from '@ant-design/icons';
import {
  listExperiments,
  getExperimentDetail,
  generatePlots,
  downloadExperimentImages,
  downloadExperimentAllFiles,
} from '../../api/client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

// 指标中文映射
const METRIC_LABELS: Record<string, string> = {
  accuracy: '准确率',
  asr: '攻击成功率',
  loss: '损失',
  precision: '精确率',
  recall: '召回率',
  fpr: '假正例率',
  auc: 'AUC',
  tp: '真阳性',
  fp: '假阳性',
  fn: '假阴性',
  tn: '真阴性',
  avg_malicious_removal_rate: '平均恶意移除率',
  avg_benign_removal_rate: '平均良性移除率',
  avg_kept_malicious: '平均保留恶意数',
  avg_kept_benign: '平均保留良性数',
  last_round: '最终轮次',
  dataset: '数据集',
};

interface ExperimentSummary {
  name: string;
  accuracy: number | null;
  asr: number | null;
  loss: number | null;
  last_round: number;
  precision?: number;
  recall?: number;
  fpr?: number;
  auc?: number;
  dataset?: string;
  created?: string;
}

const ExperimentHistory: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedExp, setSelectedExp] = useState<string | null>(null);
  const [detailData, setDetailData] = useState<any>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [plotGenerating, setPlotGenerating] = useState<string | null>(null);

  const loadExperiments = async () => {
    setLoading(true);
    try {
      const res = await listExperiments();
      setExperiments(res.data);
    } catch (error) {
      message.error('加载实验列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadExperiments();
  }, []);

  const handleViewDetail = async (name: string) => {
    setSelectedExp(name);
    setDetailModalVisible(true);
    setDetailLoading(true);
    try {
      const res = await getExperimentDetail(name);
      setDetailData(res.data);
    } catch (error) {
      message.error('加载实验详情失败');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleGeneratePlots = async (expName: string) => {
    setPlotGenerating(expName);
    try {
      await generatePlots(expName);
      message.success('图表生成成功');
      if (selectedExp === expName) {
        const res = await getExperimentDetail(expName);
        setDetailData(res.data);
      }
    } catch (error) {
      message.error('图表生成失败');
    } finally {
      setPlotGenerating(null);
    }
  };

  const handleDownloadZip = async (
    fetchFn: () => Promise<any>,
    defaultFilename: string,
    onSuccess?: () => void
  ) => {
    try {
      const res = await fetchFn();
      const blob = new Blob([res.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = defaultFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      onSuccess?.();
    } catch (error: any) {
      if (error.response?.status === 404) {
        message.error('文件不存在');
      } else {
        message.error('下载失败');
      }
    }
  };

  const handleDownloadImages = async (exp: ExperimentSummary) => {
    try {
      await downloadExperimentImages(exp.name);
      // 如果成功，直接下载
      await handleDownloadZip(
        () => downloadExperimentImages(exp.name),
        `${exp.name}_images.zip`
      );
    } catch (error: any) {
      if (error.response?.status === 404) {
        Modal.confirm({
          title: '该实验暂无图表',
          content: '是否先生成图表？',
          okText: '生成并下载',
          cancelText: '取消',
          onOk: async () => {
            try {
              await generatePlots(exp.name);
              message.success('图表生成成功，开始下载');
              await handleDownloadZip(
                () => downloadExperimentImages(exp.name),
                `${exp.name}_images.zip`
              );
            } catch (err) {
              message.error('图表生成失败');
            }
          },
        });
      } else {
        message.error('下载失败');
      }
    }
  };

  const getDownloadMenuItems = (exp: ExperimentSummary): MenuProps['items'] => [
    {
      key: 'main-csv',
      label: '主日志 CSV',
      icon: <FileTextOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        window.open(`http://localhost:8000/results/${exp.name}/${exp.name}.csv`);
      },
    },
    {
      key: 'clients-csv',
      label: '客户端日志 CSV',
      icon: <FileTextOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        window.open(`http://localhost:8000/results/${exp.name}/${exp.name}_clients.csv`);
      },
    },
    {
      key: 'metrics-csv',
      label: 'Metrics 日志 CSV',
      icon: <FileTextOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        window.open(`http://localhost:8000/results/${exp.name}/${exp.name}_metrics.csv`);
      },
    },
    {
      key: 'run-log',
      label: '运行日志',
      icon: <FileTextOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        window.open(`http://localhost:8000/results/${exp.name}/run.log`);
      },
    },
    {
      type: 'divider',
    },
    {
      key: 'images-zip',
      label: '下载图表 (ZIP)',
      icon: <FileImageOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        handleDownloadImages(exp);
      },
    },
    {
      key: 'all-files-zip',
      label: '下载全部文件 (ZIP)',
      icon: <FileZipOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        handleDownloadZip(
          () => downloadExperimentAllFiles(exp.name),
          `${exp.name}_all_files.zip`
        );
      },
    },
  ];
  const columns = [
    { title: '指标', dataIndex: 'metric', key: 'metric' },
    {
      title: '中文',
      dataIndex: 'metric',
      key: 'label',
      render: (metric: string) => METRIC_LABELS[metric] || metric,
    },
    { title: '值', dataIndex: 'value', key: 'value' },
  ];

  const metricTableData = detailData?.metrics
    ? Object.entries(detailData.metrics)
        .filter(([key]) => !key.startsWith('curve_data'))
        .map(([key, value]) => ({ metric: key, value: String(value) }))
    : [];

  const downloadFile = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <>
      <Card title="历史实验" extra={<Button onClick={loadExperiments}>刷新</Button>}>
        <Spin spinning={loading}>
          <Row gutter={[16, 16]}>
            {experiments.map((exp) => (
              <Col key={exp.name} xs={24} sm={12} md={8} lg={6}>
                <Card
                  hoverable
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleViewDetail(exp.name)}
                  actions={[
                    <Tooltip title="生成图表" key="plot">
                      <BarChartOutlined
                        onClick={(e) => {
                          e.stopPropagation();
                          handleGeneratePlots(exp.name);
                        }}
                      />
                    </Tooltip>,
                    <Tooltip title="下载文件" key="download">
                      <Dropdown
                        menu={{ items: getDownloadMenuItems(exp) }}
                        trigger={['click']}
                      >
                        <DownloadOutlined onClick={(e) => e.stopPropagation()} />
                      </Dropdown>
                    </Tooltip>,
                  ]}
                >
                  <div
                    style={{
                      fontSize: 16,
                      fontWeight: 500,
                      marginBottom: 8,
                      wordBreak: 'break-word',
                    }}
                  >
                    {exp.name}
                  </div>
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="ACC"
                        value={
                          exp.accuracy !== null
                            ? `${(exp.accuracy * 100).toFixed(2)}%`
                            : 'N/A'
                        }
                        prefix={<ExperimentOutlined />}
                        valueStyle={{ fontSize: 20 }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="ASR"
                        value={
                          exp.asr !== null
                            ? `${(exp.asr * 100).toFixed(2)}%`
                            : 'N/A'
                        }
                        prefix={<AimOutlined />}
                        valueStyle={{ fontSize: 20 }}
                      />
                    </Col>
                  </Row>
                  <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                    <div>Loss: {exp.loss?.toFixed(4) || 'N/A'}</div>
                    <div>数据集: {exp.dataset || 'N/A'}</div>
                    <div>
                      AUC: {exp.auc?.toFixed(3) || 'N/A'} | 轮次: {exp.last_round}
                    </div>
                    {exp.created && <div>创建: {exp.created.replace('T', ' ').substring(0, 22)}</div>}
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </Spin>
      </Card>

      <Modal
        title={`实验详情: ${selectedExp}`}
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={1100}
      >
        <Spin spinning={detailLoading}>
          {detailData && (
            <>
              <Row gutter={16} style={{ marginBottom: 24 }}>
                <Col span={12}>
                  <Statistic
                    title="准确率 (ACC)"
                    value={
                      detailData.metrics.accuracy != null
                        ? `${(detailData.metrics.accuracy * 100).toFixed(2)}%`
                        : 'N/A'
                    }
                    prefix={<ExperimentOutlined />}
                    valueStyle={{ fontSize: 32 }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="攻击成功率 (ASR)"
                    value={
                      detailData.metrics.asr != null
                        ? `${(detailData.metrics.asr * 100).toFixed(2)}%`
                        : 'N/A'
                    }
                    prefix={<AimOutlined />}
                    valueStyle={{ fontSize: 32 }}
                  />
                </Col>
              </Row>
              <Descriptions size="small" column={4} style={{ marginBottom: 16 }}>
                <Descriptions.Item label="Loss">
                  {detailData.metrics.loss?.toFixed(4) || 'N/A'}
                </Descriptions.Item>
                <Descriptions.Item label="数据集">
                  {detailData.metrics.dataset || 'N/A'}
                </Descriptions.Item>
                <Descriptions.Item label="最终轮次">
                  {detailData.metrics.last_round}
                </Descriptions.Item>
                <Descriptions.Item label="AUC">
                  {detailData.metrics.auc?.toFixed(3) || 'N/A'}
                </Descriptions.Item>
              </Descriptions>

              {detailData.metrics?.curve_data && (
                <div style={{ marginBottom: 24 }}>
                  <h4>ACC / ASR 曲线</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={detailData.metrics.curve_data.round.map(
                        (r: number, i: number) => ({
                          round: r,
                          accuracy: detailData.metrics.curve_data.accuracy[i],
                          asr: detailData.metrics.curve_data.asr[i],
                        })
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#8884d8"
                        name="ACC"
                      />
                      <Line
                        type="monotone"
                        dataKey="asr"
                        stroke="#82ca9d"
                        name="ASR"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              <h4>指标汇总</h4>
              <Table
                columns={columns}
                dataSource={metricTableData}
                pagination={false}
                size="small"
                style={{ marginBottom: 16 }}
              />

              <Space style={{ marginBottom: 16 }}>
                <Button
                  icon={<BarChartOutlined />}
                  onClick={() => handleGeneratePlots(selectedExp!)}
                  loading={plotGenerating === selectedExp}
                >
                  生成图表
                </Button>
                <Button
                  icon={<FileTextOutlined />}
                  onClick={() =>
                    window.open(
                      `http://localhost:8000/results/${selectedExp}/${selectedExp}.csv`
                    )
                  }
                >
                  下载 CSV
                </Button>
              </Space>

              {detailData.images && detailData.images.length > 0 && (
                <>
                  <Divider />
                  <h4>生成图表</h4>
                  <Image.PreviewGroup>
                    <Row gutter={[16, 16]}>
                      {detailData.images.map((img: string) => (
                        <Col span={12} key={img}>
                          <Card
                            size="small"
                            title={img}
                            extra={
                              <Button
                                size="small"
                                icon={<DownloadOutlined />}
                                onClick={() =>
                                  downloadFile(
                                    `http://localhost:8000/results/${selectedExp}/${img}`,
                                    img
                                  )
                                }
                              />
                            }
                          >
                            <Image
                              src={`http://localhost:8000/results/${selectedExp}/${img}`}
                              alt={img}
                              style={{ width: '100%', cursor: 'pointer' }}
                            />
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  </Image.PreviewGroup>
                </>
              )}
            </>
          )}
        </Spin>
      </Modal>
    </>
  );
};

export default ExperimentHistory;