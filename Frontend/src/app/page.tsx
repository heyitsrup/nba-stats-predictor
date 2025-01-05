import React from 'react';

import PlayerForm from '../components/PlayerForm'
import PlayerTable from '../components/PlayerTable'
import Navbar from '../components/Navbar';

const Home: React.FC = () => {
  return (
    <main className="min-h-screen flex items-center justify-center bg-slate-700">
      <div>
        <Navbar />
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
