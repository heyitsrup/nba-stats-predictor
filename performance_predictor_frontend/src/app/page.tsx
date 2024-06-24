import React from 'react';
import Image from 'next/image';

import PlayerForm from '../components/PlayerForm'
import PlayerTable from '../components/PlayerTable'

const Home: React.FC = () => {
  return (
    <main className="min-h-screen flex items-center justify-center bg-slate-700">
      <div className="w-screen m-auto fixed top-0 z-50 flex justify-center items-center space-x-3 bg-black rounded-b-lg">
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
        <p className="text-5xl text-white">StatsPredictor</p>
      </div>
      <div className="w-full flex flex-col items-center mt-80">
        <div className="flex justify-center">
          <PlayerForm />
        </div>
        <PlayerTable />
      </div>
    </main>

  );
};

export default Home;
