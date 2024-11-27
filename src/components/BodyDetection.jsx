import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

const BodyDetection = () => {
  const [isModelLoaded, setModelLoaded] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      await tf.setBackend("webgl");
      await tf.ready();
      // Load MoveNet model from TF Hub
      const modelUrl = "https://tfhub.dev/google/movenet/singlepose/lightning/4";
      const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });
      setModelLoaded(true);
      console.log("Model loaded!");
      detectPose(model);
    };

    const detectPose = async (model) => {
      if (videoRef.current && isModelLoaded) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        const detect = async () => {
          const input = tf.browser.fromPixels(video);
          const predictions = await model.executeAsync(input);
          
          // Draw the detected poses
          drawPose(predictions, ctx);
          input.dispose();
          requestAnimationFrame(detect);
        };

        detect();
      }
    };

    loadModel();
  }, [isModelLoaded]);

  const drawPose = (predictions, ctx) => {
    // You can process the predictions here and draw them on canvas
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    predictions[0].forEach((keypoint) => {
      const { y, x } = keypoint;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    });
  };

  return (
    <div className="App">
      <h2>Pose Detection with MoveNet</h2>
      <video
        ref={videoRef}
        width="640"
        height="480"
        autoPlay
        muted
        controls
        onPlay={() => setModelLoaded(false)} // Make sure model loads only when video plays
      >
        <source src="/my-new-app/public/videoplayback.mp4" type="video/mp4" />
      </video>
      <canvas ref={canvasRef} width="640" height="480" />
    </div>
  );
};

export default BodyDetection;
