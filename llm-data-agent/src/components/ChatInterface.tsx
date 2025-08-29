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

  // Parse Markdown table and remove table text from content
  const parseMarkdownTable = (text: string): { columns: string[], rows: any[][], cleanedContent: string } | null => {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    // Check if the response contains a table (starts with "|")
    const tableStartIndex = lines.findIndex(line => line.startsWith('|'));
    if (tableStartIndex === -1) return null;

    // Collect table lines
    let tableLines = [];
    let tableEndIndex = tableStartIndex;
    for (let i = tableStartIndex; i < lines.length; i++) {
      if (lines[i].startsWith('|')) {
        tableLines.push(lines[i]);
        tableEndIndex = i;
      } else {
        break;
      }
    }
    if (tableLines.length < 3) return null; // Need header, separator, and at least one row

    // Extract headers
    const headerLine = tableLines[0];
    const columns = headerLine
      .split('|')
      .map(col => col.trim())
      .filter(col => col !== '');

    // Validate separator line (e.g., "|--------|------|")
    if (!tableLines[1].match(/^\|[-|:\s]+$/)) return null;

    // Extract rows
    const rows = tableLines.slice(2).map(line =>
      line
        .split('|')
        .map(cell => cell.trim())
        .filter(cell => cell !== '')
    );

    // Ensure rows match column count
    if (rows.length === 0 || rows.some(row => row.length !== columns.length)) return null;

    // Remove table text and any preceding line (e.g., "Mean values for numeric columns:")
    const cleanedContent = [
      ...lines.slice(0, tableStartIndex > 0 ? tableStartIndex - 1 : tableStartIndex),
      ...lines.slice(tableEndIndex + 1)
    ].join('\n').trim();

    return { columns, rows, cleanedContent };
  };

  // Parse summary statistics and remove stats text from content
  const parseSummaryStats = (text: string): { columns: string[], rows: any[][], cleanedContent: string } | null => {
    // Look for the summary statistics section
    const statsMatch = text.match(/Summary statistics: ([^;]+;)+/);
    if (!statsMatch) return null;

    const statsStr = statsMatch[0].replace('Summary statistics: ', '');
    const stats = statsStr.split(';').filter(s => s.trim() !== '');
    if (stats.length === 0) return null;

    // Define columns for summary statistics
    const columns = ['Column', 'Mean', 'Std', 'Min', 'Max'];

    // Parse each stat into a row
    const rows = stats.map(stat => {
      const match = stat.match(/(\w+): mean=([\d.]+), std=([\d.]+), min=([\d.]+), max=([\d.]+)/);
      if (!match) return null;
      const [, column, mean, std, min, max] = match;
      return [column, mean, std, min, max];
    }).filter(row => row !== null) as any[][];

    if (rows.length === 0) return null;

    // Remove the summary statistics portion from the content
    const cleanedContent = text.replace(statsMatch[0], '').trim();

    return { columns, rows, cleanedContent };
  };

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
        const answer = response.data.answer;

        // Try parsing as a Markdown table
        let tableData = parseMarkdownTable(answer);
        let content = tableData ? tableData.cleanedContent : answer;

        // If no Markdown table and the query includes "eda analysis", try parsing summary statistics
        if (!tableData && input.toLowerCase().includes('eda analysis')) {
          tableData = parseSummaryStats(answer);
          content = tableData ? tableData.cleanedContent : answer;
        }

        // Add message with table or content
        if (tableData) {
          addMessage({ 
            role: 'assistant', 
            content: content.trim() !== '' ? content : undefined, 
            table: { columns: tableData.columns, rows: tableData.rows }
          });
        } else {
          addMessage({ role: 'assistant', content: answer });
        }
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