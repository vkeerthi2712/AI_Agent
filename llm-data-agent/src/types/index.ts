export interface ChatMessage {
  role: 'user' | 'assistant';
  content?: string;
  image?: string;
  table?: {
    columns: string[];
    rows: any[][];
  };
}

export interface DatasetInfo {
  dataset_id: string;
  columns: string[];
}

export interface QAResponse {
  answer: string | { columns: string[]; rows: any[][] };
}