'use client'

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import "../app/globals.css";

const PlayerForm: React.FC = () => {
  const [playerName, setPlayerName] = useState<string>('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/process-player-data/', { player_name: playerName });
      console.log(playerName)
      console.log(response.data);
      router.push(`/player/${playerName}`);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className='flex items-center space-x-3 mt-[-300px]'>
      <label className='flex items-center space-x-2'>
        <span className="whitespace-nowrap text-white">Enter player name:</span>
        <input
          type="text"
          id="text-input"
          className="mt-1 block w-full rounded-md py-2 px-4"
          value={playerName}
          onChange={(e) => setPlayerName(e.target.value)}
          required
        />
      </label>
      <button type="submit" className="mt-2 px-4 py-2 bg-nba-red text-white rounded-md hover:scale-110 transition">
        Submit
      </button>
    </form>
  );
};

export default PlayerForm;
