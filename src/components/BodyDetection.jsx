import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { moveNet } from "@tensorflow-models/posenet";

const BodyDetection = () => {
  const [isModelLoaded, setModelLoaded] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      // Load MoveNet model from TF Hub
      const model = await moveNet.load(); // Load the model once
      setModelLoaded(true);
      console.log("MoveNet Model loaded!");
      detectPose(model); // Start detecting pose once the model is loaded
    };

    const detectPose = async (model) => {
      if (videoRef.current && isModelLoaded) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        const detect = async () => {
          const input = tf.browser.fromPixels(video);
          const poses = await model.estimateSinglePose(input, {
            flipHorizontal: false,
          });

          // Draw the detected poses on canvas
          drawPose(poses, ctx);
          input.dispose(); // Dispose of the tensor to avoid memory leaks
          requestAnimationFrame(detect); // Continuously detect poses
        };

        detect();
      }
    };

    loadModel(); // Load the model when the component mounts
  }, [isModelLoaded]);

  const drawPose = (pose, ctx) => {
    // Clear the canvas before drawing the new pose
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Draw the keypoints (red circles)
    pose.keypoints.forEach((keypoint) => {
      const { y, x, score } = keypoint;
      if (score > 0.5) { // Only draw keypoints with confidence above 50%
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
      }
    });

    // Optionally, draw skeleton lines between the keypoints
    drawSkeleton(pose.keypoints, ctx);
  };

  const drawSkeleton = (keypoints, ctx) => {
    const adjacentKeyPoints = [
      ["leftShoulder", "leftElbow"],
      ["leftElbow", "leftWrist"],
      ["rightShoulder", "rightElbow"],
      ["rightElbow", "rightWrist"],
      ["leftHip", "leftKnee"],
      ["leftKnee", "leftAnkle"],
      ["rightHip", "rightKnee"],
      ["rightKnee", "rightAnkle"],
      ["leftShoulder", "rightShoulder"],
      ["leftHip", "rightHip"],
    ];

    adjacentKeyPoints.forEach(([part1, part2]) => {
      const keypoint1 = keypoints.find((p) => p.part === part1);
      const keypoint2 = keypoints.find((p) => p.part === part2);

      if (keypoint1 && keypoint2 && keypoint1.score > 0.5 && keypoint2.score > 0.5) {
        const { x: x1, y: y1 } = keypoint1.position;
        const { x: x2, y: y2 } = keypoint2.position;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = "green";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  };

  return (
    <div className="flex flex-col items-center mt-8">
      <h2 className="text-2xl font-semibold mb-4">Pose Detection with MoveNet</h2>
      <div className="relative">
        {/* Video with no controls (auto play) */}
        <video
          ref={videoRef}
          className="w-full max-w-xl rounded-lg border-2 border-gray-300 mb-4"
          autoPlay
          muted
          controls={false} // Removed video controls
          onPlay={() => setModelLoaded(false)} // Only load model when the video starts
        >
          <source src="/my-new-app/public/videoplayback.mp4" type="video/mp4" />
        </video>
        
        {/* Canvas for drawing pose */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full max-w-xl h-full rounded-lg"
        />
      </div>
    </div>
  );
};

export default BodyDetection;

