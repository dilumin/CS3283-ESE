import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const Graph = ({ data }) => {
  // Ensure the data object is valid
  if (!data || typeof data !== 'object') {
    return <div>No valid data provided</div>;
  }

  // Extract keys (x-axis values) and values (y-axis values) from the object
  const xValues = Object.keys(data); // X-axis: numbers
  const yValues = Object.values(data); // Y-axis: frequencies

  const chartData = {
    labels: xValues, // X-axis labels
    datasets: [
      {
        label: 'Frequency',
        data: yValues, // Y-axis data points
        backgroundColor: 'rgba(75, 192, 192, 0.2)', // Bar fill color
        borderColor: 'rgba(75, 192, 192, 1)', // Bar border color
        borderWidth: 1, // Border width
      },
    ],
  };

  const options = {
    scales: {
      x: {
        title: {
          display: true,
          text: 'Crack Width in Pixels', // X-axis title
        },
      },
      y: {
        title: {
          display: true,
          text: 'Frequency', // Y-axis title
        },
        beginAtZero: true, // Start Y-axis at zero
      },
    },
    plugins: {
      legend: {
        display: true,
        position: 'top', // Display legend at the top
      },
      title: {
        display: true,
        text: 'Number vs Frequency', // Chart title
      },
    },
  };

  return (
    <div style={{ width: '700px', height: '350px' }}> {/* Set desired width and height */}
      <Bar data={chartData} options={options} />
    </div>
  );
};

export default Graph;