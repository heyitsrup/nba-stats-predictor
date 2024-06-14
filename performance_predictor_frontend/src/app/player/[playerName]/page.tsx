// src/app/player/[playerName].tsx
'use client'

import React from 'react';
import { useParams } from 'next/navigation';

const PlayerPage: React.FC = () => {
  const { playerName } = useParams();

  return (
    <div>
      <h1>Player Page</h1>
      <p>Player Name: {playerName}</p>
    </div>
  );
};

export default PlayerPage;
