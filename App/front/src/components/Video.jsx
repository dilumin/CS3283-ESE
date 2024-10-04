import React, { useEffect, useState, useRef } from 'react';

const Video = () => {
  const [perpendicularData, setPerpendicularData] = useState([]);
  const canvasRef = useRef(null);

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
    }
  };

  const calculateFrequency = (data) => {
    const frequencyMap = {};
    data.forEach((distance) => {
      frequencyMap[distance] = (frequencyMap[distance] || 0) + 1;
    });
    return frequencyMap;
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
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  // Function to draw the bar graph
  const drawGraph = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frequencyData = calculateFrequency(perpendicularData);
    const distances = Object.keys(frequencyData);
    const frequencies = Object.values(frequencyData);

    // Graph properties
    const barWidth = 40;
    const barSpacing = 20;
    const graphHeight = 200;
    const maxFrequency = Math.max(...frequencies, 1);

    // Adjust the canvas width dynamically based on the number of bars
    const canvasWidth = (distances.length * (barWidth + barSpacing)) + 100;
    canvas.width = canvasWidth;

    // Draw bars
    distances.forEach((distance, index) => {
      const x = index * (barWidth + barSpacing) + 50;
      const barHeight = (frequencies[index] / maxFrequency) * graphHeight;
      const y = canvas.height - barHeight - 30;

      // Draw bar
      ctx.fillStyle = 'rgba(75, 192, 192, 0.6)';
      ctx.fillRect(x, y, barWidth, barHeight);

      // Draw labels for distances (x-axis)
      ctx.fillStyle = '#000';
      ctx.fillText(distance, x + barWidth / 4, canvas.height - 10);

      // Draw frequency values above bars
      ctx.fillText(frequencies[index], x + barWidth / 4, y - 5);
    });

    // Draw y-axis label
    ctx.fillText('Frequency', 10, 20);
  };

  useEffect(() => {
    if (perpendicularData.length > 0) {
      drawGraph();
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


      {/* Display the scrollable canvas for the graph */}
      <div style={styles.scrollContainer}>
        <canvas ref={canvasRef} height="300" style={styles.canvas} />
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
    width: '80%',
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