import React, { useState } from 'react';
import MainLayout from './components/MainLayout';
import { type ChatMessage, type DatasetInfo } from './types';
import './styles/styles.css';

const App: React.FC = () => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [dataset, setDataset] = useState<any | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);

  const addMessage = (message: ChatMessage) => {
    setChatHistory((prev) => [...prev, message]);
  };

  const handleDatasetUpload = (info: DatasetInfo | null, data: any | null) => {
    setDatasetInfo(info);
    setDataset(data);
  };

  return (
    <MainLayout
      datasetInfo={datasetInfo}
      dataset={dataset}
      chatHistory={chatHistory}
      error={error}
      setDatasetInfo={handleDatasetUpload}
      addMessage={addMessage}
      setError={setError}
    />
  );
};

export default App;