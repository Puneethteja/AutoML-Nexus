import React, { useState } from 'react';
import axios from 'axios';
import { Play, CheckCircle2, AlertCircle, Loader2, FileSpreadsheet, RotateCcw } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]); 
  const [targetColumn, setTargetColumn] = useState("");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  const resetDashboard = () => {
    setStatus("idle");
    setProgress(0);
    setResult(null);
    setErrorMessage("");
    setFile(null);
    setColumns([]);
    setTargetColumn("");
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const headers = text.split('\n')[0].split(',').map(h => h.trim());
      setColumns(headers);
    };
    reader.readAsText(selectedFile);
  };

  const handleUpload = async () => {
    if (!file || !targetColumn) return alert("Please select a file and a target column.");

    resetDashboard();
    setStatus("processing");
    setProgress(10);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_column", targetColumn);

    try {
      const response = await axios.post("http://localhost:8000/start-training", formData);
      pollStatus(response.data.task_id);
    } catch (err) {
      setStatus("failed");
      setErrorMessage(err.response?.data?.detail || "Failed to start pipeline.");
    }
  };

  const pollStatus = (taskId) => {
    const timer = setInterval(async () => {
      try {
        const res = await axios.get(`http://localhost:8000/status/${taskId}`);
        setProgress(res.data.progress || 50);

        if (res.data.status === "completed") {
          setStatus("completed");
          setResult(res.data.result);
          clearInterval(timer);
        } else if (res.data.status === "failed") {
          throw new Error(res.data.error || "Training failed on server.");
        }
      } catch (err) {
        setStatus("failed");
        setErrorMessage(err.message);
        clearInterval(timer);
      }
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-6">
      <div className="max-w-3xl mx-auto bg-white shadow-sm rounded-xl p-8 border border-gray-200">
        <div className="flex items-center space-x-3 mb-8">
          <div className="p-3 bg-blue-50 text-blue-600 rounded-lg"><FileSpreadsheet size={28} /></div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">AutoML-Nexus</h1>
            <p className="text-sm text-gray-500">AutoML Nexus | System Analytics</p>
          </div>
        </div>

        <div className="space-y-6">
          <input 
            type="file" 
            accept=".csv"
            onChange={handleFileChange} 
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer border border-gray-200 rounded-lg" 
          />

          {columns.length > 0 && (
            <select 
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full px-4 py-3 border border-gray-200 rounded-lg text-sm bg-white focus:ring-2 focus:ring-blue-500 outline-none"
            >
              <option value="">Select target column...</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          )}

          <button 
            onClick={status === "completed" || status === "failed" ? resetDashboard : handleUpload} 
            className={`w-full flex items-center justify-center text-white font-medium py-3 rounded-lg transition ${status === "processing" ? "bg-gray-400" : "bg-blue-600 hover:bg-blue-700"}`}
          >
            {status === "processing" ? <><Loader2 className="animate-spin mr-2" /> Processing...</> : 
             status === "completed" || status === "failed" ? <><RotateCcw className="mr-2" /> Run New Pipeline</> : 
             <><Play className="mr-2" /> Run ML Pipeline</>}
          </button>

          {status !== "idle" && status !== "completed" && status !== "failed" && (
            <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
              <div className="bg-blue-600 h-2.5 transition-all duration-500" style={{ width: `${progress}%` }}></div>
            </div>
          )}

          {status === "failed" && (
            <div className="flex items-center p-4 bg-red-50 text-red-700 rounded-lg border border-red-200 text-sm">
              <AlertCircle className="mr-2" /> {errorMessage}
            </div>
          )}

          {status === "completed" && result && (
            <div className="dashboard-card">
              <h2 className="text-xl font-bold mb-4">System Validation Summary</h2>
              <div className="metric-grid">
                <div className="metric-box">
                  <p className="text-xs text-gray-500">Train Acc</p>
                  <strong className="text-lg">{(result.train_accuracy * 100).toFixed(2)}%</strong>
                </div>
                <div className="metric-box">
                  <p className="text-xs text-gray-500">Test Acc</p>
                  <strong className="text-lg">{(result.test_accuracy * 100).toFixed(2)}%</strong>
                </div>
                <div className="metric-box">
                  <p className="text-xs text-gray-500">Std Dev</p>
                  <strong className="text-lg">{result.std_dev.toFixed(4)}</strong>
                </div>
              </div>
              <div className="mt-6">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">Run Manifest</h3>
                <pre className="json-box">{JSON.stringify(result, null, 2)}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;