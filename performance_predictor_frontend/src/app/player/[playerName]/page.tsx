'use client'

import React, { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Image from 'next/image';

const PlayerPage: React.FC = () => {
  const { playerName } = useParams();

  const [prediction, setPrediction] = useState<number[]>([]);

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

  if (prediction === null) {
    return <div>Loading...</div>;
  }

  const decodedPlayerName = decodeURIComponent(String(playerName));
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-700">
      <div className="w-auto m-auto rounded-2xl fixed top-5 p-3 z-50 flex items-center space-x-3">
        <Image 
          src='/PlayerTable/Luka_Doncic.jpg' 
          alt='logo' 
          width={150}
          height={150}
          className="rounded-full"
        />
        <p className="text-5xl text-white">{decodedPlayerName}</p>
      </div>

      <table className="w-screen table-fixed text-white">
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
    </div>
  );
};

export default PlayerPage;
