'use client'

import React from 'react';
import Image from 'next/image';
import axios from 'axios';
import { useRouter } from 'next/navigation';
import "../app/globals.css";

const PlayerTable: React.FC = () => {
    const router = useRouter();

    const handleImageClick = async (playerName: string) => {
        try {
            const response = await axios.post('http://127.0.0.1:8000/api/process-player-data/', { player_name: playerName });
            console.log(response.data);
            router.push(`/player/${playerName}`);
        } catch (error) {
            console.error('Error:', error);
        }
    };
    return (
        <table className="w-screen table-fixed">
            <tr className='w-1/5'>
                <td>
                    <Image 
                        src='/PlayerTable/Nikola_Jokic.jpg' 
                        alt='Nikola_Jokic' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-4 mb-8"
                        onClick={() => handleImageClick('Nikola Jokic')}
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Jimmy_Butler.jpg' 
                        alt='Jimmy_Butler' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-4 mb-8"
                        onClick={() => handleImageClick('Jimmy Butler')}
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Luka_Doncic.jpg' 
                        alt='Luka_Doncic' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-4 mb-8"
                        onClick={() => handleImageClick('Luka Doncic')}
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Anthony_Edwards.jpg' 
                        alt='Anthony_Edwards' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-4 mb-8"
                        onClick={() => handleImageClick('Anthony Edwards')}
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Giannis_Antetokounmpo.jpeg' 
                        alt='Giannis_Antetokounmpo' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-4 mb-8"
                        onClick={() => handleImageClick('Giannis Antetokounmpo')}
                    />
                </td>
            </tr>
            <tr className='w-1/4'>
                <td>
                    <Image 
                        src='/PlayerTable/LeBron_James.jpeg' 
                        alt='LeBron_James' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-44"
                        onClick={() => handleImageClick('LeBron James')}
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Kevin_Durant.jpg' 
                        alt='Kevin_Durant' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-44"
                        onClick={() => handleImageClick('Kevin Durant')}
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Stephen_Curry.jpg' 
                        alt='Stephen_Curry' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-44"
                        onClick={() => handleImageClick('Stephen Curry')}
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Kawhi_Leonard.jpg' 
                        alt='Kawhi_Leonard' 
                        width={225}
                        height={225}
                        className="rounded-full hover:scale-110 transition cursor-pointer ml-44"
                        onClick={() => handleImageClick('Kawhi Leonard')}
                    />
                </td>
            </tr>
        </table>
    );
};

export default PlayerTable;