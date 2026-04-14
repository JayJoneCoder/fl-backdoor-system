import React, { useEffect, useState } from 'react';
import {
  Card,
  Button,
  Table,
  Modal,
  Checkbox,
  message,
  Space,
  Input,
  Image,
  Row,
  Col,
  Spin,
  Divider,
  Dropdown,
} from 'antd';
import type { MenuProps } from 'antd';
import {
  FileTextOutlined,
  PlusOutlined,
  BarChartOutlined,
  CodeOutlined,
  DownloadOutlined,
  FileZipOutlined,
  FileImageOutlined,
} from '@ant-design/icons';
import {
  listExperiments,
  listSummaries,
  createSummary,
  generateSummaryPlots,
  downloadSummaryZip,
  getSummaryDetail,
} from '../api/client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

interface ExperimentItem {
  name: string;
}

interface SummaryItem {
  name: string;
  experiments: string[];
  created: string | null;
}

const SummaryPage: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentItem[]>([]);
  const [summaries, setSummaries] = useState<SummaryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedExps, setSelectedExps] = useState<string[]>([]);
  const [summaryName, setSummaryName] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [detailData, setDetailData] = useState<any>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [plotGenerating, setPlotGenerating] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    try {
      const [expRes, sumRes] = await Promise.all([listExperiments(), listSummaries()]);
      setExperiments(expRes.data);
      setSummaries(sumRes.data);
    } catch (error) {
      message.error('加载数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleCreateSummary = async () => {
    if (selectedExps.length === 0) {
      message.warning('请至少选择一个实验');
      return;
    }
    setSubmitting(true);
    try {
      await createSummary({
        experiments: selectedExps,
        output_name: summaryName || undefined,
      });
      message.success('总结任务已提交');
      setModalVisible(false);
      setSelectedExps([]);
      setSummaryName('');
      loadData();
    } catch (error) {
      message.error('创建总结失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleGeneratePlots = async (summaryName: string) => {
    setPlotGenerating(summaryName);
    try {
      await generateSummaryPlots(summaryName);
      message.success('图表生成成功');
      if (detailData && detailData.name === summaryName) {
        const res = await getSummaryDetail(summaryName);
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

  const handleDownloadZip = async (summaryName: string) => {
    try {
      const res = await downloadSummaryZip(summaryName);
      const blob = new Blob([res.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${summaryName}_images.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      message.success('开始下载压缩包');
    } catch (error) {
      message.error('下载失败');
    }
  };

  const handleDownloadImages = async (summary: SummaryItem) => {
    try {
      await downloadSummaryZip(summary.name);
      await handleDownloadZip(summary.name);
    } catch (error: any) {
      if (error.response?.status === 404) {
        Modal.confirm({
          title: '该总结暂无图表',
          content: '是否先生成图表？',
          okText: '生成并下载',
          cancelText: '取消',
          onOk: async () => {
            try {
              await generateSummaryPlots(summary.name);
              message.success('图表生成成功，开始下载');
              await handleDownloadZip(summary.name);
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

  const handleViewDetail = async (summaryName: string) => {
    setDetailLoading(true);
    setDetailModalVisible(true);
    try {
      const res = await getSummaryDetail(summaryName);
      setDetailData(res.data);
    } catch (error) {
      message.error('加载详情失败');
      setDetailModalVisible(false);
    } finally {
      setDetailLoading(false);
    }
  };

  const getDownloadMenuItems = (record: SummaryItem): MenuProps['items'] => [
    {
      key: 'csv',
      label: '总结 CSV',
      icon: <FileTextOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        window.open(`http://localhost:8000/results/summaries/${record.name}/summary.csv`);
      },
    },
    {
      key: 'latex',
      label: 'LaTeX 表格',
      icon: <CodeOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        window.open(`http://localhost:8000/results/summaries/${record.name}/summary_table.tex`);
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
        handleDownloadImages(record);
      },
    },
    {
      key: 'all-zip',
      label: '下载全部文件 (ZIP)',
      icon: <FileZipOutlined />,
      onClick: (e) => {
        e.domEvent.stopPropagation();
        handleDownloadZip(record.name);
      },
    },
  ];

  const columns = [
    { title: '总结名称', dataIndex: 'name', key: 'name', width: 200 },
    {
      title: '包含实验',
      dataIndex: 'experiments',
      key: 'experiments',
      render: (list: string[]) => list?.join(', ') || '',
    },
    { title: '创建时间', dataIndex: 'created', key: 'created', width: 200 },
    {
      title: '操作',
      key: 'action',
      width: 200,
      render: (_: any, record: SummaryItem) => (
        <Space wrap size="small">
          <Button
            size="small"
            icon={<BarChartOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              handleGeneratePlots(record.name);
            }}
            loading={plotGenerating === record.name}
          >
            生成图表
          </Button>
          <Dropdown menu={{ items: getDownloadMenuItems(record) }} trigger={['click']}>
            <Button size="small" icon={<DownloadOutlined />} onClick={(e) => e.stopPropagation()}>
              下载
            </Button>
          </Dropdown>
        </Space>
      ),
    },
  ];

  const prepareChartData = () => {
    if (!detailData?.experiments_curve_data) return [];
    const expNames = Object.keys(detailData.experiments_curve_data);
    if (expNames.length === 0) return [];
    const firstExp = expNames[0];
    const rounds = detailData.experiments_curve_data[firstExp].round;
    const data = rounds.map((r: number, idx: number) => {
      const point: any = { round: r };
      expNames.forEach((name) => {
        point[`${name}_acc`] = detailData.experiments_curve_data[name].accuracy[idx];
        point[`${name}_asr`] = detailData.experiments_curve_data[name].asr[idx];
      });
      return point;
    });
    return data;
  };

  const chartData = prepareChartData();
  const expNames = detailData ? Object.keys(detailData.experiments_curve_data || {}) : [];

  return (
    <div style={{ width: '100%' }}>
      <h2>实验总结</h2>
      <Card>
        <Space style={{ marginBottom: 16 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
            新建总结
          </Button>
          <Button onClick={loadData}>刷新</Button>
        </Space>
        <Table
          dataSource={summaries}
          columns={columns}
          rowKey="name"
          loading={loading}
          scroll={{ x: 1000 }}
          onRow={(record) => ({
            style: { cursor: 'pointer' },
            onClick: () => handleViewDetail(record.name),
          })}
        />
      </Card>

      <Modal
        title="选择实验并命名总结"
        open={modalVisible}
        onOk={handleCreateSummary}
        onCancel={() => setModalVisible(false)}
        confirmLoading={submitting}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Input
            placeholder="总结名称（可选，默认时间戳）"
            value={summaryName}
            onChange={(e) => setSummaryName(e.target.value)}
          />
          <Checkbox.Group
            options={experiments.map((e) => ({ label: e.name, value: e.name }))}
            value={selectedExps}
            onChange={(values) => setSelectedExps(values as string[])}
          />
        </Space>
      </Modal>

      <Modal
        title={`总结详情: ${detailData?.name}`}
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={1200}
      >
        <Spin spinning={detailLoading}>
          {detailData && (
            <>
              {expNames.length > 0 && (
                <>
                  <h4>所有实验 ACC 曲线对比</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      {expNames.map((name) => (
                        <Line
                          key={name}
                          type="monotone"
                          dataKey={`${name}_acc`}
                          name={name}
                          stroke={`#${Math.floor(Math.random() * 16777215).toString(16)}`}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>

                  <h4 style={{ marginTop: 24 }}>所有实验 ASR 曲线对比</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      {expNames.map((name) => (
                        <Line
                          key={name}
                          type="monotone"
                          dataKey={`${name}_asr`}
                          name={name}
                          stroke={`#${Math.floor(Math.random() * 16777215).toString(16)}`}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                  <Divider />
                </>
              )}

              <h4>汇总表格</h4>
              <Table
                dataSource={detailData.summary_table}
                columns={Object.keys(detailData.summary_table[0] || {}).map((k) => ({ title: k, dataIndex: k, key: k }))}
                pagination={false}
                size="small"
                scroll={{ x: 1000 }}
              />

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
                                onClick={() => downloadFile(`http://localhost:8000/results/summaries/${detailData.name}/${img}`, img)}
                              />
                            }
                          >
                            <Image
                              src={`http://localhost:8000/results/summaries/${detailData.name}/${img}`}
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
    </div>
  );
};

export default SummaryPage;