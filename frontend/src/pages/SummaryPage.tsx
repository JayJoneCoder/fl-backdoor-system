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
  Tooltip,
} from 'antd';
import {
  FileTextOutlined,
  PlusOutlined,
  BarChartOutlined,
  PictureOutlined,
  CodeOutlined,
  DownloadOutlined,
  FileZipOutlined,
} from '@ant-design/icons';
import {
  listExperiments,
  listSummaries,
  createSummary,
  generateSummaryPlots,
  getSummaryImages,
  downloadSummaryZip,
} from '../api/client';

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

  const [previewVisible, setPreviewVisible] = useState(false);
  const [currentSummary, setCurrentSummary] = useState<string | null>(null);
  const [images, setImages] = useState<string[]>([]);
  const [imagesLoading, setImagesLoading] = useState(false);
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
      if (currentSummary === summaryName) {
        fetchImages(summaryName);
      }
    } catch (error) {
      message.error('图表生成失败');
    } finally {
      setPlotGenerating(null);
    }
  };

  const fetchImages = async (summaryName: string) => {
    setImagesLoading(true);
    try {
      const res = await getSummaryImages(summaryName);
      setImages(res.data.images);
    } catch (error) {
      message.error('加载图片失败');
    } finally {
      setImagesLoading(false);
    }
  };

  const openImagePreview = (summaryName: string) => {
    setCurrentSummary(summaryName);
    setPreviewVisible(true);
    fetchImages(summaryName);
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

  const columns = [
    { title: '总结名称', dataIndex: 'name', key: 'name', width: 180 },
    {
      title: '包含实验',
      dataIndex: 'experiments',
      key: 'experiments',
      render: (list: string[]) => list?.join(', ') || '',
    },
    { title: '创建时间', dataIndex: 'created', key: 'created', width: 180 },
    {
      title: '操作',
      key: 'action',
      width: 400,
      render: (_: any, record: SummaryItem) => (
        <Space wrap size="small">
          <Button
            size="small"
            icon={<FileTextOutlined />}
            onClick={() => window.open(`http://localhost:8000/results/summaries/${record.name}/summary.csv`)}
          >
            CSV
          </Button>
          <Button
            size="small"
            icon={<CodeOutlined />}
            onClick={() => window.open(`http://localhost:8000/results/summaries/${record.name}/summary_table.tex`)}
          >
            LaTeX
          </Button>
          <Button
            size="small"
            icon={<BarChartOutlined />}
            onClick={() => handleGeneratePlots(record.name)}
            loading={plotGenerating === record.name}
          >
            生成图表
          </Button>
          <Button
            size="small"
            icon={<PictureOutlined />}
            onClick={() => openImagePreview(record.name)}
          >
            查看
          </Button>
          <Button
            size="small"
            icon={<FileZipOutlined />}
            onClick={() => handleDownloadZip(record.name)}
          >
            打包下载
          </Button>
        </Space>
      ),
    },
  ];

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
        title={`总结图表 - ${currentSummary}`}
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={null}
        width={1100}
      >
        <Spin spinning={imagesLoading}>
          {images.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>
              暂无图表，请点击“生成图表”按钮
            </div>
          ) : (
            <>
              <Divider />
              <Image.PreviewGroup>
                <Row gutter={[16, 16]}>
                  {images.map((img) => (
                    <Col span={12} key={img}>
                      <Card
                        size="small"
                        title={img}
                        extra={
                          <Tooltip title="下载图片">
                            <Button
                              size="small"
                              icon={<DownloadOutlined />}
                              onClick={() => downloadFile(`http://localhost:8000/results/summaries/${currentSummary}/${img}`, img)}
                            />
                          </Tooltip>
                        }
                      >
                        <Image
                          src={`http://localhost:8000/results/summaries/${currentSummary}/${img}`}
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
        </Spin>
      </Modal>
    </div>
  );
};

export default SummaryPage;