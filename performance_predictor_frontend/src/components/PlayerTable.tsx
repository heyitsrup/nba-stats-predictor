'use client'

import React from 'react';
import Image from 'next/image';
import "../app/globals.css";

const PlayerTable: React.FC = () => {
    return (
        <table className="w-screen table-fixed">
            <tr className='w-1/5'>
                <td>
                    <Image 
                        src='/PlayerTable/Nikola_Jokic.jpg' 
                        alt='Nikola_Jokic' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Jimmy_Butler.jpg' 
                        alt='Jimmy_Butler' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Luka_Doncic.jpg' 
                        alt='Luka_Doncic' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Anthony_Edwards.jpg' 
                        alt='Anthony_Edwards' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Giannis_Antetokounmpo.jpeg' 
                        alt='Giannis_Antetokounmpo' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
            </tr>
            <tr className='w-1/4'>
                <td>
                    <Image 
                        src='/PlayerTable/Lebron_James.jpeg' 
                        alt='Lebron_James' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Kevin_Durant.jpg' 
                        alt='Kevin_Durant' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Stephen_Curry.jpg' 
                        alt='Stephen_Curry' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
                <td>
                    <Image 
                        src='/PlayerTable/Kawhi_Leonard.jpg' 
                        alt='Kawhi_Leonard' 
                        width={150}
                        height={150}
                        className="rounded-full hover:scale-110 transition"
                    />
                </td>
            </tr>
        </table>
    );
};

export default PlayerTable;