import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [feedback, setFeedback] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Upload file");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      setResult(res.data);
      setLoading(false);
    } catch (err) {
      alert("Error connecting to backend");
      setLoading(false);
    }
  };

  const sendFeedback = async () => {
    await axios.post("http://127.0.0.1:8000/feedback", {
      prediction: result.prediction,
      feedback: feedback,
    });

    alert("Feedback submitted");
    setFeedback("");
  };

  return (
    <div className="container">
      <h1 className="title">COGNIFY</h1>
      <p className="subtitle">
        Understand your lecture better. Analyze cognitive load with AI.
      </p>

      <div className="card">
        <h2>Upload Lecture Audio</h2>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload}>Analyze Lecture</button>
      </div>

      {loading && <p className="loading">⚡ Processing audio...</p>}

      {result && (
        <div className="results">
          <h2>
            Prediction:{" "}
            <span className="highlight">
              {result.prediction?.toUpperCase()}
            </span>
          </h2>

          <h3>Probability:</h3>
          <p>Low: {result.probability?.low?.toFixed(3)}</p>
          <p>Medium: {result.probability?.medium?.toFixed(3)}</p>
          <p>High: {result.probability?.high?.toFixed(3)}</p>

          <h3>Summary:</h3>
          <p>{result.summary}</p>

          <h3>Quiz:</h3>
          <ul>
            {result.quiz?.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>

          <h3>Give Feedback</h3>
          <input
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Enter feedback"
          />
          <button onClick={sendFeedback}>Submit Feedback</button>
        </div>
      )}
    </div>
  );
}

export default App;