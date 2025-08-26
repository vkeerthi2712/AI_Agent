import React, { useState } from 'react';
import axios, { AxiosError } from 'axios';
import { type ChatMessage } from '../types';

interface ChatInterfaceProps {
  datasetId: string | null;
  addMessage: (message: ChatMessage) => void;
  setError: (error: string | null) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ datasetId, addMessage, setError }) => {
  const [input, setInput] = useState('');

  const handleSubmit = async () => {
    if (!input.trim()) {
      setError('Please enter a message.');
      return;
    }
    if (!datasetId) {
      setError('Please upload a dataset first.');
      return;
    }

    addMessage({ role: 'user', content: input });

    const plotKeywords = [
      'plot',
      'chart',
      'graph',
      'heatmap',
      'scatter',
      'line',
      'bar',
      'histogram',
      'box',
      'violin',
      'swarm',
      'pie',
      'volcano',
      'tga',
      'dtg',
      'cluster',
      'curve',
      'regression',
      'kaplan',
    ];
    const isPlotRequest = plotKeywords.some((keyword) => input.toLowerCase().includes(keyword));

    try {
      if (isPlotRequest) {
        const response = await axios.post(
          'http://127.0.0.1:8000/plot',
          { dataset_id: datasetId, instruction: input, format: 'png' },
          { responseType: 'arraybuffer' }
        );
        const base64Image = `data:image/png;base64,${btoa(
          new Uint8Array(response.data).reduce((data, byte) => data + String.fromCharCode(byte), '')
        )}`;
        addMessage({ role: 'assistant', content: 'Generated Plot', image: base64Image });
      } else {
        const response = await axios.post('http://127.0.0.1:8000/ask', {
          dataset_id: datasetId,
          question: input,
        });
        addMessage({ role: 'assistant', content: response.data.answer });
      }
      setError(null);
      setInput('');
    } catch (err) {
      const error = err as AxiosError<{ detail?: string }>;
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error';
      addMessage({ role: 'assistant', content: `Error: ${errorMessage}` });
      setError(`Request failed: ${errorMessage}`);
    }
  };

  return (
    <div className="chat-container">
      <h2>Chat with the Data Agent</h2>
      <p>Ask questions about the dataset or request plots (e.g., 'Plot a heatmap of the correlation matrix', 'What are the columns?', 'Show summary statistics').</p>
      <div className="chat-input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g., Plot a heatmap or What is the mean of age?"
          onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
        />
        <button onClick={handleSubmit}>Send</button>
      </div>
    </div>
  );
};

export default ChatInterface;