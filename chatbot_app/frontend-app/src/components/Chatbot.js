import React, { useState } from 'react';
import axios from 'axios';
import './Chatbot.css';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [context, setContext] = useState([]);
  const maxContextLength = 5;

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const updateContext = (userInput, botResponse) => {
    const newContext = [...context, { user: userInput, bot: botResponse }];
    if (newContext.length > maxContextLength) {
      newContext.shift();
    }
    setContext(newContext);
  };

  const handleSend = async () => {
    if (inputText.trim() !== '' && !isLoading) {
      const userMessage = { id: messages.length + 1, text: `user: ${inputText}`, sender: 'user' };
      setMessages([...messages, userMessage]);
      setInputText('');
      setIsLoading(true);
      
      try {
        const response = await axios.post('http://127.0.0.1:5000/chatbot', {
          user_input: inputText,
          context
        });
        const botResponse = response.data.chatbot_response;
        const botMessage = { id: messages.length + 2, text: `bot: ${botResponse}`, sender: 'bot' };
        setMessages((msgs) => [...msgs, botMessage]);
        updateContext(inputText, botResponse);
      } catch (error) {
        console.error('Error fetching data:', error);
        setMessages((msgs) => [...msgs, { id: messages.length + 2, text: 'Bot: I am having trouble connecting. Please try again later.', sender: 'bot' }]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="chatbot-container">
      <h1>Mental Health Support Chatbot</h1>
      <div className="messages-container">
        {messages.length === 0 && !isLoading ? (
          <div className="message start-conversation">Let's begin the conversation !!!</div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}`}>
              {message.text}
            </div>
          ))
        )}
        {isLoading && (<div className="loading-spinner"></div>)}
      </div>
      <div className="input-container">
        <input
          type="text"
          value={inputText}
          onChange={handleInputChange}
          onKeyPress={(event) => { if (event.key === 'Enter') handleSend(); }}
          disabled={isLoading}
        />
        <button onClick={handleSend} disabled={isLoading}>Send</button>
      </div>
    </div>
  );
}

export default Chatbot;
