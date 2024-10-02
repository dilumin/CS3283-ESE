import React, { useEffect, useState } from 'react';

const Video = () => {
  const [perpendicularData, setPerpendicularData] = useState([]);

  // Fetch perpendicular distances from YOLO feed
  const fetchPerpendicularData = () => {
    const yoloFeed = new EventSource('http://127.0.0.1:5001/yolo');

    yoloFeed.onmessage = function (event) {
      try {
        const contentType = event.data.match(/Content-Type:\s*([^;]+)/)[1];
        
        if (contentType === 'application/json') {
          const distances = JSON.parse(event.data);  // If the response is JSON (distances)
          setPerpendicularData(distances);
        }
      } catch (error) {
        console.error("Error processing YOLO data: ", error);
      }
    };
    
    yoloFeed.onerror = function (error) {
      console.error('Error with YOLO feed:', error);
      yoloFeed.close();
    };

    return () => {
      yoloFeed.close();
    };
  };

  useEffect(() => {
    fetchPerpendicularData();
  }, []);

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

      {/* Display perpendicular distance data */}
      <div style={styles.dataContainer}>
        <h2>Perpendicular Distances:</h2>
        {perpendicularData.length === 0 ? (
          <p>No distance data available.</p>
        ) : (
          <ul>
            {perpendicularData.map((distance, index) => (
              <li key={index}>Distance {index + 1}: {distance} cm</li>
            ))}
          </ul>
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
  dataContainer: {
    marginTop: '20px',
    textAlign: 'center',
  },
};

export default Video;