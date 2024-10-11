// import logo from './logo.svg';
// import './App.css';
import Video from './components/Video';

// function App() {
//   return (
//     <div className="App">
//       
//     </div>
//   );
// }

// export default App;

import React, { useEffect, useState } from 'react';
import Graph from './components/Graph';

const App = () => {
  const [cleanedData, setCleanedData] = useState({});

  useEffect(() => {
    // Example data { 1: 1, 2: 2, 3: 1 }
    const exampleData = { 1: 1, 2: 2, 3: 1 };
    setCleanedData(exampleData);
  }, []);

  return (
    <div className="App">
      <Video />

    </div>
  );
};

export default App;
