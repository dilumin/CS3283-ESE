import React from 'react';

const sendStartStop = (val) => {
    fetch('http://127.0.0.1:5001/startstop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',  // This ensures Flask recognizes the JSON data
        },
        body: JSON.stringify({
            start_stop: val
        }),
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.log(error));
};

const Video = () => {
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
            alt="Live Video Feed"
            style={styles.video}
            />
        </div>

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
  con: {
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
};

export default Video;