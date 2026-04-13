import React from 'react';
import { Input, InputNumber, Select, Switch } from 'antd';

interface FieldRendererProps {
  fieldType: string;
  options?: string[];
  placeholder?: string;
  value?: any;
  onChange?: (value: any) => void;
}

const FieldRenderer: React.FC<FieldRendererProps> = ({
  fieldType,
  options,
  placeholder,
  value,
  onChange,
}) => {
  const commonProps = {
    style: { width: '100%' },
    placeholder,
    value,
    onChange,
  };

  switch (fieldType) {
    case 'string':
      if (options && options.length > 0) {
        return (
          <Select {...commonProps}>
            {options.map((opt) => (
              <Select.Option key={opt} value={opt}>
                {opt}
              </Select.Option>
            ))}
          </Select>
        );
      }
      return <Input {...commonProps} />;

    case 'integer':
      return <InputNumber {...commonProps} />;

    case 'float':
      return <InputNumber {...commonProps} step={0.01} />;

    case 'boolean':
      return <Switch checked={value} onChange={onChange} />;

    default:
      return <Input {...commonProps} />;
  }
};

export default FieldRenderer;