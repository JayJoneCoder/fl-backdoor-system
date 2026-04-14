import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// 获取配置 schema
export const getConfigSchema = () => apiClient.get('/api/config/schema');

// 获取当前配置
export const getConfig = () => apiClient.get('/api/config');

// 更新配置
export const updateConfig = (updates: Record<string, any>) =>
  apiClient.post('/api/config', updates);

// 启动实验
export const startExperiment = (name: string, config?: Record<string, any>) =>
  apiClient.post('/api/experiment/start', { name, config });

// 停止实验
export const stopExperiment = () => apiClient.post('/api/experiment/stop');

// 获取实验状态
export const getExperimentStatus = () => apiClient.get('/api/experiment/status');

// 获取实验列表
export const listExperiments = () => apiClient.get('/api/experiments');

// 获取实验详情
export const getExperimentDetail = (name: string) =>
  apiClient.get(`/api/experiments/${name}`);

// 解析批量实验 JSON
export const parseBatchConfig = (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return apiClient.post('/api/batch/parse', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

// 运行批量实验
export const runBatchExperiments = (experiments: any[]) =>
  apiClient.post('/api/batch/run', { experiments });

// 上传整个 toml 文件（高级模式）
export const uploadConfigFile = (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return apiClient.post('/api/config/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

// 为指定实验生成图表
export const generatePlots = (expName: string) =>
  apiClient.post(`/api/experiments/${expName}/plot`);

// 生成所有实验的总结报告
export const generateSummary = () =>
  apiClient.post('/api/experiments/summarize');

export const listSummaries = () => apiClient.get('/api/summaries');
export const createSummary = (data: { experiments: string[]; output_name?: string }) =>
  apiClient.post('/api/summarize', data);

// 为总结生成图表
export const generateSummaryPlots = (summaryName: string) =>
  apiClient.post(`/api/summaries/${summaryName}/plot`);

// 获取总结的图片列表
export const getSummaryImages = (summaryName: string) =>
  apiClient.get(`/api/summaries/${summaryName}/images`);

// 下载总结 ZIP 包
export const downloadSummaryZip = (summaryName: string) =>
  apiClient.get(`/api/summaries/${summaryName}/download`, { responseType: 'blob' });

// 下载实验所有图片 ZIP
export const downloadExperimentImages = (expName: string) =>
  apiClient.get(`/api/experiments/${expName}/images/download`, { responseType: 'blob' });

// 下载实验全部文件 ZIP
export const downloadExperimentAllFiles = (expName: string) =>
  apiClient.get(`/api/experiments/${expName}/all/download`, { responseType: 'blob' });

// 获取总结详情
export const getSummaryDetail = (summaryName: string) =>
  apiClient.get(`/api/summaries/${summaryName}`);