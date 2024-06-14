import React from 'react';
import Image from 'next/image';

import PlayerForm from '../components/PlayerForm'
import PlayerTable from '../components/PlayerTable'

const Home: React.FC = () => {
  return (
    <main className="min-h-screen flex items-center justify-center bg-neutral-200">
      <div className="w-auto m-auto rounded-2xl fixed top-5 p-3 z-50 flex items-center space-x-3">
        <Image 
          src='/nba.png' 
          alt='logo' 
          width={75}
          height={75}
        />
        <Image 
          src='/chart.png' 
          alt='logo' 
          width={75}
          height={75}
        />
        <p className="text-5xl">StatsPredictor</p>
      </div>
      <div className="w-full flex flex-col items-center mt-60">
        <div className="w-full flex justify-center">
          <PlayerForm />
        </div>
        <PlayerTable />
      </div>
    </main>
  );
};

export default Home;
