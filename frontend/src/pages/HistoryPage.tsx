import React from 'react';
import ExperimentHistory from '../components/ExperimentHistory';

const HistoryPage: React.FC = () => {
  return (
    <div>
      <h2 className="page-title">历史实验记录</h2>
      <ExperimentHistory />
    </div>
  );
};

export default HistoryPage;