import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Play, Loader2, FileSpreadsheet, RotateCcw, Download, CheckCircle2 } from 'lucide-react';
import OptimizationChart from './components/OptimizationChart';

function App() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]); 
  const [targetColumn, setTargetColumn] = useState("");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  
  const fileInputRef = useRef(null);

  const resetDashboard = () => {
    setStatus("idle");
    setProgress(0);
    setResult(null);
    setFile(null);
    setColumns([]);
    setTargetColumn("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDownload = () => {
    if (!result) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `results_${new Date().getTime()}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    const reader = new FileReader();
    reader.onload = (event) => {
      const headers = event.target.result.split('\n')[0].split(',').map(h => h.trim());
      setColumns(headers);
    };
    reader.readAsText(selectedFile);
  };

  const handleUpload = async () => {
    if (!file || !targetColumn) return alert("Please select a file and target column.");
    setStatus("processing");
    setProgress(10);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_column", targetColumn);
    try {
      const response = await axios.post("http://localhost:8000/start-training", formData);
      pollStatus(response.data.task_id);
    } catch (err) {
      setStatus("idle");
      alert("Failed to start pipeline.");
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
          throw new Error();
        }
      } catch (err) {
        setStatus("idle");
        clearInterval(timer);
      }
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-6 font-sans text-gray-800">
      <div className="max-w-xl mx-auto bg-white shadow-xl rounded-2xl p-8 border border-gray-100">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-extrabold text-gray-900">AutoML-Nexus</h1>
          <p className="text-gray-500 text-sm">Pipeline Execution Engine</p>
        </header>

        <div className="space-y-6">
          <input type="file" ref={fileInputRef} accept=".csv" onChange={handleFileChange} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer border rounded-full p-1" />
          
          {columns.length > 0 && (
            <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} className="w-full px-4 py-3 border rounded-xl bg-gray-50 focus:ring-2 focus:ring-blue-500 outline-none">
              <option value="">Select Target Column</option>
              {columns.map(col => <option key={col} value={col}>{col}</option>)}
            </select>
          )}

          {status === "idle" && (
            <button onClick={handleUpload} className="w-full bg-blue-600 text-white font-bold py-3 rounded-xl hover:bg-blue-700 transition flex items-center justify-center">
              <Play size={18} className="mr-2" /> Start Training
            </button>
          )}

          {status === "processing" && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm font-bold text-blue-600"><span>Processing Data...</span><span>{progress}%</span></div>
              <div className="w-full bg-gray-200 rounded-full h-2.5"><div className="bg-blue-600 h-2.5 rounded-full transition-all" style={{ width: `${progress}%` }}></div></div>
            </div>
          )}

          {status === "completed" && (
            <div className="text-center space-y-6 animate-in fade-in duration-500">
              <div className="p-4 bg-green-50 text-green-700 rounded-xl flex items-center justify-center font-bold">
                <CheckCircle2 className="mr-2" /> Training Complete
              </div>
              
              {result?.history && <OptimizationChart data={result.history} />}
              
              <div className="flex gap-4">
                <button onClick={handleDownload} className="flex-1 bg-blue-600 text-white font-bold py-3 rounded-xl hover:bg-blue-700 flex items-center justify-center">
                  <Download size={18} className="mr-2" /> Download Full Results
                </button>
                <button onClick={resetDashboard} className="bg-gray-100 px-6 py-3 rounded-xl hover:bg-gray-200">
                  <RotateCcw size={18} />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;