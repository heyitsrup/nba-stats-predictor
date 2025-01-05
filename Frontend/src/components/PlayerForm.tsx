'use client'

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import Loading from './Loading';
import "../app/globals.css";

const PlayerForm: React.FC = () => {
  const [playerName, setPlayerName] = useState<string>('');
  const [isLoading, setLoading] = useState<boolean>(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:5000/api/process-player-data/', { player_name: playerName });
      router.push(`/player/${playerName}`);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false); // Reset loading state
    }
  };

  return (
    <div>
      {isLoading && <Loading />}
      <form onSubmit={handleSubmit} className='flex items-center space-x-3 mt-[-150px]'>
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
    </div>
  );
};

export default PlayerForm;
