import React from 'react';
import { type ChatMessage } from '../types';

interface ChatHistoryProps {
  messages: ChatMessage[];
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ messages }) => {
  return (
    <div className="chat-history">
      <h2>Chat History</h2>
      <div>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong>
            {msg.content && <p>{msg.content}</p>}
            {msg.image && (
              <div>
                <p>{msg.content}</p>
                <img src={msg.image} alt="Generated Plot" />
                <a href={msg.image} download="generated_plot.png">💾 Download Plot</a>
              </div>
            )}
            {msg.table && (
              <table className="chat-table">
                <thead>
                  <tr>
                    {msg.table.columns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {msg.table.rows.map((row: any[], rowIndex: number) => (
                    <tr key={rowIndex}>
                      {row.map((cell, cellIndex) => (
                        <td key={cellIndex}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatHistory;