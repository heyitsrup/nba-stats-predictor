import React from 'react';
import Image from 'next/image';

import PlayerForm from '../components/PlayerForm'

const Home: React.FC = () => {
  return (
    <main className="min-h-screen flex items-center justify-center bg-neutral-200">
      <div className="w-auto m-auto rounded-2xl fixed top-5 p-3 z-50 flex items-center space-x-3">
        <Image 
          src='/nba.png' 
          alt='logo' 
          width={75} // Provide appropriate width
          height={75} // Provide appropriate height
        />
        <Image 
          src='/chart.png' 
          alt='logo' 
          width={75} // Provide appropriate width
          height={75} // Provide appropriate height
        />
        <p className="text-5xl">StatsPredictor</p>
      </div>
      <PlayerForm />
    </main>
  );
};

export default Home;
