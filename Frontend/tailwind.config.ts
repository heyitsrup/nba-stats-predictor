import type { Config } from "tailwindcss";

const config: Config = {
  // content: [
  //   "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
  //   "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  //   "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  // ],
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'nba-red': '#cc152e',
        'nba-blue': '#24438b',
        'nba-white': '#000000'
      },
  }
  },
  plugins: [],
};
export default config;
