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
} from 'antd';
import {
  BarChartOutlined,
  DownloadOutlined,
  FileTextOutlined,
  ExperimentOutlined,
  AimOutlined,
} from '@ant-design/icons';
import {
  listExperiments,
  getExperimentDetail,
  generatePlots,
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
  created?: string;   // 实验创建时间
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

  const downloadFile = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

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
                      <BarChartOutlined onClick={(e) => { e.stopPropagation(); handleGeneratePlots(exp.name); }} />
                    </Tooltip>,
                    <Tooltip title="下载 CSV" key="csv">
                      <FileTextOutlined onClick={(e) => { e.stopPropagation(); window.open(`http://localhost:8000/results/${exp.name}/${exp.name}.csv`); }} />
                    </Tooltip>,
                  ]}
                >
                  {/* 新增：实验名称置顶 */}
                  <div style={{ fontSize: 16, fontWeight: 500, marginBottom: 8, wordBreak: 'break-word' }}>
                    {exp.name}
                  </div>
                  <Row gutter={8}>
                    <Col span={12}>
                      <Statistic
                        title="ACC"
                        value={exp.accuracy !== null ? `${(exp.accuracy * 100).toFixed(2)}%` : 'N/A'}
                        prefix={<ExperimentOutlined />}
                        valueStyle={{ fontSize: 20 }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="ASR"
                        value={exp.asr !== null ? `${(exp.asr * 100).toFixed(2)}%` : 'N/A'}
                        prefix={<AimOutlined />}
                        valueStyle={{ fontSize: 20 }}
                      />
                    </Col>
                  </Row>
                  <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                    <div>Loss: {exp.loss?.toFixed(4) || 'N/A'}</div>
                    <div>数据集: {exp.dataset || 'N/A'}</div>
                    <div>AUC: {exp.auc?.toFixed(3) || 'N/A'} | 轮次: {exp.last_round}</div>
                    {exp.created && <div>创建: {exp.created}</div>}
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
                    value={detailData.metrics.accuracy != null ? `${(detailData.metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}
                    prefix={<ExperimentOutlined />}
                    valueStyle={{ fontSize: 32 }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="攻击成功率 (ASR)"
                    value={detailData.metrics.asr != null ? `${(detailData.metrics.asr * 100).toFixed(2)}%` : 'N/A'}
                    prefix={<AimOutlined />}
                    valueStyle={{ fontSize: 32 }}
                  />
                </Col>
              </Row>
              <Descriptions size="small" column={4} style={{ marginBottom: 16 }}>
                <Descriptions.Item label="Loss">{detailData.metrics.loss?.toFixed(4) || 'N/A'}</Descriptions.Item>
                <Descriptions.Item label="数据集">{detailData.metrics.dataset || 'N/A'}</Descriptions.Item>
                <Descriptions.Item label="最终轮次">{detailData.metrics.last_round}</Descriptions.Item>
                <Descriptions.Item label="AUC">{detailData.metrics.auc?.toFixed(3) || 'N/A'}</Descriptions.Item>
              </Descriptions>

              {detailData.metrics?.curve_data && (
                <div style={{ marginBottom: 24 }}>
                  <h4>ACC / ASR 曲线</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={detailData.metrics.curve_data.round.map((r: number, i: number) => ({
                        round: r,
                        accuracy: detailData.metrics.curve_data.accuracy[i],
                        asr: detailData.metrics.curve_data.asr[i],
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Line type="monotone" dataKey="accuracy" stroke="#8884d8" name="ACC" />
                      <Line type="monotone" dataKey="asr" stroke="#82ca9d" name="ASR" />
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
                  onClick={() => window.open(`http://localhost:8000/results/${selectedExp}/${selectedExp}.csv`)}
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
                                onClick={() => downloadFile(`http://localhost:8000/results/${selectedExp}/${img}`, img)}
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