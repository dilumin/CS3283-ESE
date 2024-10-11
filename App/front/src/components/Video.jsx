import React, { useEffect, useState, useRef } from 'react';
import Graph from './Graph';

const Video = () => {
  const [perpendicularData, setPerpendicularData] = useState([]);
  const canvasRef = useRef(null);
  const [cleanedData, setCleanedData] = useState({});

  const fetchData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5001/data', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json(); // Parse the JSON data
      console.log(data); // Log the fetched data to the console
      setPerpendicularData(data); // Update the state with the fetched data
    } catch (error) {
      console.error('Error:', error);
      //for testing
      // setPerpendicularData([1, 2,2, 3, 4, 5, 6, 7, 8,8,8, 9,9,9,9,9, 10,10,10, 11, 12, 13, 14, 15, 16,16,16, 17, 18, 19, 20,25 ,15 , 17]);
    }
  };


  const resetData = async () => {
    try{
      await fetch('http://127.0.0.1:5001/reset', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    }
    catch (error) {
      console.error('Error:', error);
    }
    setPerpendicularData([]);
    setCleanedData({});

  }

  function generateClusterData(data) {
    // Step 1: Find the key with the highest value
    let maxKey = null;
    let maxValue = -Infinity;
    let indexMax = 0;

    const keys = Object.keys(data);
    const values = Object.values(data);

    for (let i = 0; i < keys.length; i++) {
      if (values[i] > maxValue) {
        indexMax = i;
        maxKey = keys[i];
        maxValue = values[i];
      }
    }
    const start = Math.max(0, indexMax - 3); 
    const end = Math.min(keys.length, indexMax + 4);
    const clusterData = {};
    for (let i = start; i < end; i++) {
      clusterData[keys[i]] = values[i];
    }
    return clusterData;
  }



  const cleanData = (perpendicularData) =>{
    perpendicularData.sort();
    var cleanedData = perpendicularData.reduce((acc, curr) => {
      acc[curr] = (acc[curr] || 0) + 1;
      return acc; 
    },{});
    
    cleanedData = generateClusterData(cleanedData);
    console.log(cleanedData);
    setCleanedData(cleanedData);
    
  }


  useEffect(() => {
    if (perpendicularData.length > 0) {

      cleanData(perpendicularData);

    }
  }, [perpendicularData]);

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Live Video Feed</h1>
      <div style={styles.con}>
        <div style={styles.videoContainer}>
          <img
            src="http://127.0.0.1:5001/video_feed"
            alt="Live Video Feed"
            style={styles.video}
          />
        </div>
        <div style={styles.videoContainer}>
          <img
            src="http://127.0.0.1:5001/yolo"
            alt="YOLO Processed Feed"
            style={styles.video}
          />
        </div>
      </div>

      <button onClick={fetchData}>Get Data</button>
      <button onClick={resetData}>Reset</button>


      <h1>No of Pixels X Crack Widths  </h1>
      <div >
      {Object.keys(cleanedData).length > 0 && (
        <Graph data={cleanedData} />
      )}

      </div>

    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    backgroundColor: '#f0f0f0',
  },
  title: {
    fontSize: '24px',
    color: '#333',
    marginBottom: '20px',
  },
  con: {
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoContainer: {
    width: '100%',
    maxWidth: '700px',
    height: '400px',
    border: '2px solid #ccc',
    borderRadius: '10px',
    overflow: 'hidden',
    backgroundColor: '#000',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: '20px',
    margin: '10px',
  },
  video: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  scrollContainer: {
    width: '80%',
    maxWidth: '700px',
    marginTop: '20px',
    overflowX: 'auto', // Make the container scrollable horizontally
  },
  canvas: {
    border: '1px solid #ccc',
  },
};

export default Video;