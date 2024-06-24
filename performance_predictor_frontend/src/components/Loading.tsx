'use client'

const Loading = () => {
    return (
      <div className="fixed top-0 left-0 z-50 w-full h-full overflow-hidden bg-nba-white bg-opacity-75 flex items-center justify-center">
        <div className="flex items-center">
          <svg className="animate-spin h-10 w-10 text-white mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span className="text-white text-xl">Loading...</span>
        </div>
      </div>
    );
  };
  
  export default Loading;
  
  