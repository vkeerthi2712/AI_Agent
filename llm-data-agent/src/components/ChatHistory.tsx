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
            {msg.image ? (
              <div>
                <p>{msg.content}</p>
                <img src={msg.image} alt="Generated Plot" />
                <a href={msg.image} download="generated_plot.png">💾 Download Plot</a>
              </div>
            ) : (
              <p>{msg.content}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatHistory;