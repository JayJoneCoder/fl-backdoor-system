import React from 'react';
import ExperimentHistory from '../components/ExperimentHistory';

const HistoryPage: React.FC = () => {
  return (
    <div className="workspace-page">
      <div className="page-heading">
        <h2 className="page-title">历史实验记录</h2>
        <p className="page-description">查看实验结果、运行指标与配置快照，管理和导出实验文件。</p>
      </div>
      <ExperimentHistory />
    </div>
  );
};

export default HistoryPage;