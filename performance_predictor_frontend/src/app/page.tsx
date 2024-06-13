import React from 'react';

import PlayerForm from '../components/PlayerForm'

const Home: React.FC = () => {
  return (
    <main className="min-h-screen flex items-center justify-center bg-grey">
      <PlayerForm />
    </main>
  );
};

export default Home;
