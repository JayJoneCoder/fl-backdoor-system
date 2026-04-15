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
    <div>
      <h2>实验配置</h2>
      <ConfigForm ref={configFormRef} />
      <ExperimentControl onBeforeStart={handleBeforeStart} />
    </div>
  );
};

export default ConfigPage;