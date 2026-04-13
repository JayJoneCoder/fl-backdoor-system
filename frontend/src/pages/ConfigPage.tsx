import React from 'react';
import ConfigForm from '../components/ConfigForm';
import ExperimentControl from '../components/ExperimentControl';

const ConfigPage: React.FC = () => {
  return (
    <div>
      <h2>实验配置</h2>
      <ConfigForm />
      <ExperimentControl />
    </div>
  );
};

export default ConfigPage;