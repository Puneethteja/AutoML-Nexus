import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const OptimizationChart = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-lg shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">Optimization Progress</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="trial" label={{ value: 'Trial', position: 'insideBottom', offset: -5 }} />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip />
          <Line type="monotone" dataKey="accuracy" stroke="#4f46e5" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default OptimizationChart;