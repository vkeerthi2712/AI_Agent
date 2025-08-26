import React from 'react';
import FileUpload from './FileUpload';
import ChatInterface from './ChatInterface';
import ChatHistory from './ChatHistory';
import { type ChatMessage, type DatasetInfo } from '../types';

interface MainLayoutProps {
  datasetInfo: DatasetInfo | null;
  dataset: any | null;
  chatHistory: ChatMessage[];
  error: string | null;
  setDatasetInfo: (info: DatasetInfo | null, data: any | null) => void;
  addMessage: (message: ChatMessage) => void;
  setError: (error: string | null) => void;
}

const MainLayout: React.FC<MainLayoutProps> = ({
  datasetInfo,
  dataset,
  chatHistory,
  error,
  setDatasetInfo,
  addMessage,
  setError,
}) => {
  return (
    <div className="app-container">
      <div className="left-panel">
        {error && <div className="error">{error}</div>}
        <FileUpload setDatasetInfo={setDatasetInfo} setError={setError} />
        <ChatInterface datasetId={datasetInfo?.dataset_id || null} addMessage={addMessage} setError={setError} />
      </div>
      <div className="right-panel">
        {datasetInfo && dataset && (
          <div className="dataset-info">
            <h2>Dataset Details</h2>
            <p>Rows: {dataset.shape[0]}</p>
            <p>Columns: {datasetInfo.columns.join(', ')}</p>
            <table className="dataset-table">
              <thead>
                <tr>
                  {datasetInfo.columns.map((col) => (
                    <th key={col}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset.head(5).slice(1).map((row: any[], index: number) => (
                  <tr key={index}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {chatHistory.length > 0 && <ChatHistory messages={chatHistory} />}
      </div>
    </div>
  );
};

export default MainLayout;