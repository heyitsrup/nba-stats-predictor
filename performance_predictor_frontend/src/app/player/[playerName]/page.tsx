'use client'

import React, { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';

const PlayerPage: React.FC = () => {
  const { playerName } = useParams();

  const [prediction, setPrediction] = useState<number | null>(null); // Assuming prediction is a number

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

  return (
    <div>
      <h1>Player Page</h1>
      <p>Player Name: {playerName}</p>
      <p>Prediction: {prediction}</p>
    </div>
  );
};

export default PlayerPage;
