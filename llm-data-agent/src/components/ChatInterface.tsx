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
  const [customerName, setCustomerName] = useState('');
  const [reportFormat, setReportFormat] = useState<'pdf' | 'docx'>('pdf');
  const [filename, setFilename] = useState('');

  // Parse Markdown table and remove table text from content
  const parseMarkdownTable = (text: string): { columns: string[], rows: any[][], cleanedContent: string } | null => {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    const tableSections: { columns: string[], rows: any[][], startIndex: number, endIndex: number }[] = [];

    // Find all table sections
    let i = 0;
    while (i < lines.length) {
      if (lines[i].startsWith('|')) {
        let tableLines = [];
        let tableStartIndex = i;
        let tableEndIndex = i;
        // Collect table lines
        while (i < lines.length && lines[i].startsWith('|')) {
          tableLines.push(lines[i]);
          tableEndIndex = i;
          i++;
        }
        if (tableLines.length >= 3) { // Need header, separator, and at least one row
          // Extract headers
          const headerLine = tableLines[0];
          const columns = headerLine
            .split('|')
            .map(col => col.trim())
            .filter(col => col !== '');
          
          // Validate separator line
          if (tableLines[1].match(/^\|[-|:\s]+$/)) {
            // Extract rows
            const rows = tableLines.slice(2).map(line =>
              line
                .split('|')
                .map(cell => cell.trim())
                .filter(cell => cell !== '')
            );
            // Ensure rows match column count
            if (rows.every(row => row.length === columns.length)) {
              tableSections.push({ columns, rows, startIndex: tableStartIndex, endIndex: tableEndIndex });
            }
          }
        }
      } else {
        i++;
      }
    }

    if (tableSections.length === 0) return null;

    // Combine all tables into one for display (or pick the most relevant one)
    const combinedColumns = tableSections[0].columns; // Use first table's columns for simplicity
    const combinedRows = tableSections.flatMap(section => section.rows);

    // Remove all table sections from content
    let cleanedLines = [...lines];
    // Sort sections by startIndex in descending order to avoid index shifting when removing
    const sortedSections = tableSections.sort((a, b) => b.startIndex - a.startIndex);
    for (const section of sortedSections) {
      // Remove the table lines and the line before it (e.g., section header like "**Dataset Shape**:")
      const start = section.startIndex > 0 ? section.startIndex - 1 : section.startIndex;
      cleanedLines = [
        ...cleanedLines.slice(0, start),
        ...cleanedLines.slice(section.endIndex + 1)
      ];
    }
    const cleanedContent = cleanedLines.join('\n').trim();

    return { columns: combinedColumns, rows: combinedRows, cleanedContent };
  };

  // Parse summary statistics and remove stats text from content
  const parseSummaryStats = (text: string): { columns: string[], rows: any[][], cleanedContent: string } | null => {
    // Look for the summary statistics section
    const statsMatch = text.match(/Summary Statistics for Numeric Columns:([\s\S]*?)(\n\n|$)/);
    if (!statsMatch) return null;

    const statsText = statsMatch[1].trim();
    const tableData = parseMarkdownTable(statsText);
    if (!tableData) return null;

    // Remove the summary statistics section from the content
    const cleanedContent = text.replace(statsMatch[0], '').trim();

    return {
      columns: tableData.columns,
      rows: tableData.rows,
      cleanedContent
    };
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

        // For EDA analysis or summary statistics, force table parsing
        let tableData = null;
        let content = answer;

        if (input.toLowerCase().includes('eda analysis') || input.toLowerCase().includes('summary statistics')) {
          // Try parsing as a multi-section Markdown table
          tableData = parseMarkdownTable(answer);
          content = tableData ? tableData.cleanedContent : answer;

          // If no table found for EDA/summary stats, try parsing specifically for summary statistics
          if (!tableData && input.toLowerCase().includes('summary statistics')) {
            tableData = parseSummaryStats(answer);
            content = tableData ? tableData.cleanedContent : answer;
          }
        } else {
          // Try parsing as a Markdown table for other responses
          tableData = parseMarkdownTable(answer);
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

  const handleGenerateReport = async () => {
    if (!datasetId) {
      setError('Please upload a dataset first.');
      return;
    }
    if (!customerName.trim()) {
      setError('Please enter a customer name.');
      return;
    }
    if (!filename.trim()) {
      setError('Please enter a report filename.');
      return;
    }

    try {
      const response = await axios.post(
        'http://127.0.0.1:8000/generate_report',
        { dataset_id: datasetId, customer_name: customerName, filename, format: reportFormat },
        { responseType: 'blob' }
      );

      const sanitizedFilename = filename.replace(/[^a-zA-Z0-9-_]/g, '_');
      const outputFilename = `${sanitizedFilename}.${reportFormat}`;

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', outputFilename);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      addMessage({ role: 'assistant', content: `Generated ${reportFormat.toUpperCase()} report: ${filename}` });
      setError(null);
    } catch (err) {
      const error = err as AxiosError<{ detail?: string }>;
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error';
      addMessage({ role: 'assistant', content: `Error generating report: ${errorMessage}` });
      setError(`Report generation failed: ${errorMessage}`);
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
      <div className="report-container">
        <h3>Generate Report</h3>
        <input
          type="text"
          value={customerName}
          onChange={(e) => setCustomerName(e.target.value)}
          placeholder="Enter customer name"
        />
        <input
          type="text"
          value={filename}
          onChange={(e) => setFilename(e.target.value)}
          placeholder="Enter report filename"
        />
        <select
          value={reportFormat}
          onChange={(e) => setReportFormat(e.target.value as 'pdf' | 'docx')}
        >
          <option value="pdf">PDF</option>
          <option value="docx">Word (DOCX)</option>
        </select>
        <button onClick={handleGenerateReport} disabled={!customerName || !filename || !datasetId}>
          Submit
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;
