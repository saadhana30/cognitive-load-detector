import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "https://cognitive-load-detector.onrender.com";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [feedback, setFeedback] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Upload a file first");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const res = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Backend error. Check console.");
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async () => {
    await axios.post(`${API_URL}/feedback`, {
      prediction: result.prediction,
      feedback: feedback,
    });

    alert("Feedback submitted!");
    setFeedback("");
  };

  return (
    <div className="app">
      <h1 className="logo">COGNIFY</h1>
      <p className="tagline">
        Understand your lecture better with AI-powered cognitive insights
      </p>

      {/* Upload Section */}
      <div className="card upload-card">
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload}>Analyze Lecture</button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Analyzing your lecture...</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="card results-card">
          <h2>
            Cognitive Load:{" "}
            <span className="highlight">
              {result.prediction?.toUpperCase()}
            </span>
          </h2>

          <div className="probability">
            <div>Low: {result.probability?.low?.toFixed(3)}</div>
            <div>Medium: {result.probability?.medium?.toFixed(3)}</div>
            <div>High: {result.probability?.high?.toFixed(3)}</div>
          </div>

          <div className="section">
            <h3>🧠 Summary</h3>
            <p>{result.summary}</p>
          </div>

          <div className="section">
            <h3>📚 Quiz</h3>
            <ul>
              {result.quiz?.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
          </div>

          <div className="feedback">
            <h3>💬 Feedback</h3>
            <input
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="Your thoughts..."
            />
            <button onClick={sendFeedback}>Submit</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;