'use client'

import React, { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import PlayerHeader from '../[playerName]/playerHeader'
import Navbar from '../../../components/Navbar'
import Link from 'next/link';
import { useRouter } from 'next/navigation';

const PlayerPage: React.FC = () => {
  const router = useRouter();

  const { playerName } = useParams();

  const [prediction, setPrediction] = useState<number[]>([]);

  const handleAnalyticsClick = () => {
    router.push(`/player/${playerName}/analytics`);
  };

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/predict?playerName=${playerName}`);
        if (!res.ok) {
          throw new Error('Failed to fetch prediction');
        }
        const data = await res.json();
        setPrediction(data.prediction);
      } catch (error) {
        console.error('Error fetching prediction:', error);
      }
    };

    if (playerName) {
      fetchPrediction();
    }
  }, [playerName]);
  
  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-slate-700">
      <Navbar />
      <PlayerHeader />

      <table className="w-screen table-fixed text-white mt-32">
        <tr className='text-9xl w-1/5 text-center'>
          <td>{prediction[0]}</td>
          <td>{prediction[1]}</td>
          <td>{prediction[2]}</td>
          <td>{prediction[3]}</td>
          <td>{prediction[4]}</td>
        </tr>
        <tr className='text-sm w-1/5 text-center'>
          <td>PTS</td>
          <td>REB</td>
          <td>AST</td>
          <td>STL</td>
          <td>BLK</td>
        </tr>
      </table>

      <div className="mt-10">
        <button type="button" className="text-white bg-nba-red rounded-lg px-5 py-2.5 me-4 mb-2 hover:scale-110 transition">
          <Link href="/">
            🏠 Go Home
          </Link>
        </button>
        <button type="button" className="text-white bg-nba-red rounded-lg px-5 py-2.5 me-2 mb-2 hover:scale-110 transition" onClick={handleAnalyticsClick}>📊 Analytics</button>
      </div>
    </main>
  );
};

export default PlayerPage;
