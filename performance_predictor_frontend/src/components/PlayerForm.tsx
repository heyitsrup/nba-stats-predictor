'use client'

import React, { useState } from 'react';
import axios from 'axios';

const PlayerForm: React.FC = () => {
  const [playerName, setPlayerName] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/process-player-data/', { player_name: playerName });
      console.log(response.data);
      // Handle success (optional)
    } catch (error) {
      console.error('Error:', error);
      // Handle error (optional)
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Enter player name:
        <input
          type="text"
          id="text-input"
          className="mt-1 block w-full border border-gray-300 rounded-md py-2 px-4 focus:outline-none focus:border-blue-500"
          value={playerName}
          onChange={(e) => setPlayerName(e.target.value)}
          required
        />
      </label>
      <button type="submit" className="mt-2 px-4 py-2 bg-blue-500 text-white rounded-md">
        Submit
      </button>
    </form>
  );
};

export default PlayerForm;
