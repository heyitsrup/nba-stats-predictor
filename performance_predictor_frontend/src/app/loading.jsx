import Image from 'next/image';

export default function Loading() {
    // You can add any UI inside Loading, including a Skeleton.
    return <Image 
        src='/chart.png' 
        alt='logo' 
        width={75}
        height={75}
    />
  }