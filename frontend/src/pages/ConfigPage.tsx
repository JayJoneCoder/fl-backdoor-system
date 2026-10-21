import React, { useRef } from 'react';
import ConfigForm from '../components/ConfigForm';
import type { ConfigFormHandle } from '../components/ConfigForm';
import ExperimentControl from '../components/ExperimentControl';

const ConfigPage: React.FC = () => {
  const configFormRef = useRef<ConfigFormHandle>(null);

  const handleBeforeStart = async () => {
    if (configFormRef.current) {
      return await configFormRef.current.checkUnsavedBeforeStart();
    }
    return { action: 'save' as const };
  };

  return (
    <div className="workspace-page">
      <div className="page-heading">
        <h2 className="page-title">单个实验配置</h2>
        <p className="page-description">设置联邦训练、攻击与防御参数，运行实验并查看实时指标。</p>
      </div>
      <ConfigForm ref={configFormRef} />
      <ExperimentControl onBeforeStart={handleBeforeStart} />
    </div>
  );
};

export default ConfigPage;