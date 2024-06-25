import React from 'react';
import Image from 'next/image';

const Navbar: React.FC = () => {
    return (
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
    )
}

export default Navbar;