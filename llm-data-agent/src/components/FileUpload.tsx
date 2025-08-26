import React, { useState } from 'react';
import axios, { AxiosError } from 'axios';
import { type DatasetInfo } from '../types';
import * as XLSX from 'xlsx';
import Papa from 'papaparse';

interface FileUploadProps {
  setDatasetInfo: (info: DatasetInfo | null, data: any | null) => void;
  setError: (error: string | null) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ setDatasetInfo, setError }) => {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    let dataset = null;
    try {
      if (file.name.endsWith('.csv')) {
        const text = await file.text();
        const result = Papa.parse(text, { header: true, skipEmptyLines: true });
        const rows = result.data;
        const columns = result.meta.fields || [];
        dataset = {
          shape: [rows.length, columns.length],
          head: (n: number) => [columns, ...rows.slice(0, n).map((row: any) => Object.values(row))],
          toArray: () => [columns, ...rows.map((row: any) => Object.values(row))],
        };
      } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
        const arrayBuffer = await file.arrayBuffer();
        const workbook = XLSX.read(new Uint8Array(arrayBuffer), { type: 'array' });
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        const rows = XLSX.utils.sheet_to_json(sheet, { header: 1 });
        dataset = {
          shape: [rows.length - 1, rows[0].length],
          head: (n: number) => rows.slice(0, n + 1),
          toArray: () => rows,
        };
      }

      const response = await axios.post('http://127.0.0.1:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setDatasetInfo(response.data, dataset);
      setError(null);
    } catch (err) {
      const error = err as AxiosError<{ detail?: string }>;
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error';
      setError(`Upload failed: ${errorMessage}`);
      setDatasetInfo(null, null);
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload Dataset</h2>
      <input
        type="file"
        accept=".csv,.xlsx,.xls"
        onChange={handleFileChange}
        className="file-input"
      />
      {file && <p className="file-name">Selected: {file.name}</p>}
      <button onClick={handleUpload} disabled={!file} className="upload-button">
        Upload
      </button>
    </div>
  );
};

export default FileUpload;