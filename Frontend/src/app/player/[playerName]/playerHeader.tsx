'use client';

import React, { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Image from 'next/image';
import "../../globals.css";

const PlayerHeader: React.FC = () => {
    const { playerName } = useParams();
    const decodedPlayerName = decodeURIComponent(String(playerName));
    const [playerId, setPlayerId] = useState<string | null>(null);

    useEffect(() => {
        const fetchPlayerId = async () => {
          try {
            const res = await fetch(`http://127.0.0.1:5000/api/get-player-id?playerName=${playerName}`);
            if (!res.ok) {
              throw new Error('Failed to fetch player ID');
            }
            const data = await res.json();
            setPlayerId(data.playerId);
          } catch (error) {
            console.error('Error fetching player ID:', error);
          }
        };
    
        if (playerName) {
            fetchPlayerId();
        }
      }, [playerName]);

    return (
        <div className="w-auto m-auto rounded-2xl fixed top-20 p-3 z-50 flex items-center space-x-3">
        {/* TODO: Add a default image for players without headshots */}
        <Image 
          src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${playerId}.png`}
          alt='player_headshot' 
          width={250}
          height={250}
          className="rounded-full"
        />
        <p className="text-5xl text-white">{decodedPlayerName}</p>
      </div>
    );
};

export default PlayerHeader;