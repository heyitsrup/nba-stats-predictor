'use client'

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import Navbar from '../../../../components/Navbar';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface Metrics {
    POINTS: number;
    REBOUNDS: number;
    ASSISTS: number;
    STEALS: number;
    BLOCKS: number;
}

const PlayerAnalytics: React.FC = () => {
    const router = useRouter();
    const { playerName } = useParams();

    const [predictedMetrics, setPredictedMetrics] = useState<number[][]>([]);
    const [actualMetrics, setActualMetrics] = useState<number[][]>([]);

    const handleBackClick = () => {
        router.push(`/player/${playerName}`);
    };

    useEffect(() => {
        const fetchPrediction = async () => {
            try {
                const res = await fetch(`http://127.0.0.1:8000/api/analytics?playerName=${playerName}`);
                if (!res.ok) {
                    throw new Error('Failed to fetch prediction');
                }
                const data = await res.json();
                setActualMetrics(data.actual_values);
                setPredictedMetrics(data.predicted_values);
            } catch (error) {
                console.error('Error fetching prediction:', error);
            }
        };

        if (playerName) {
            fetchPrediction();
        }
    }, [playerName]);

    const createChartData = (metricIndex: number, metricName: string) => {
        return {
            labels: actualMetrics.map((_, index) => `Game ${index + 1}`),
            datasets: [
                {
                    label: `ACTUAL ${metricName}`,
                    data: actualMetrics.map(metrics => metrics[metricIndex]),
                    borderColor: 'blue',
                    fill: false,
                },
                {
                    label: `PREDICTED ${metricName}`,
                    data: predictedMetrics.map(metrics => metrics[metricIndex]),
                    borderColor: 'red',
                    fill: false,
                },
            ],
        };
    };

    const createChartOptions = (metricName: string) => {
        return {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top' as const,
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Games',
                    },
                },
                y: {
                    title: {
                        display: true,
                        text: 'Metrics',
                    },
                },
            },
        };
    };

    return (
        <main className="min-h-screen flex flex-col items-center justify-center bg-slate-700">
            <Navbar />

            <div className="mt-32">
                <button type="button" className="text-white bg-nba-red rounded-lg px-5 py-2.5 me-4 mb-2 hover:scale-110 transition">
                    <Link href="/">
                        🏠 Go Home
                    </Link>
                </button>
                <button type="button" className="text-white bg-nba-red rounded-lg px-5 py-2.5 me-2 mb-2 hover:scale-110 transition" onClick={handleBackClick}>👤 Player Page</button>
            </div>

            <div className="w-full max-w-4xl bg-white my-16">
                {['POINTS', 'REBOUNDS', 'ASSISTS', 'STEALS', 'BLOCKS'].map((metric, index) => (
                    <div key={metric} className="my-8">
                        <Line
                            data={createChartData(index, metric)}
                            options={createChartOptions(metric)}
                        />
                    </div>
                ))}
            </div>
        </main>
    );
};

export default PlayerAnalytics;
