import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import { load as loadMoveNet } from "@tensorflow-models/movenet"; // Correct import for MoveNet model

// Function to calculate angle between three points using cosine and sine
const calculateAngle = (pointA, pointB, pointC) => {
  const dx1 = pointA.x - pointB.x;
  const dy1 = pointA.y - pointB.y;
  const dx2 = pointC.x - pointB.x;
  const dy2 = pointC.y - pointB.y;

  // Dot product and magnitudes
  const dot = dx1 * dx2 + dy1 * dy2;
  const magnitudeA = Math.sqrt(dx1 * dx1 + dy1 * dy1);
  const magnitudeB = Math.sqrt(dx2 * dx2 + dy2 * dy2);

  const cosAngle = dot / (magnitudeA * magnitudeB);
  const angle = Math.acos(cosAngle); // In radians
  return (angle * 180) / Math.PI; // Convert radians to degrees
};

const MoveNet = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [angles, setAngles] = useState({
    leftArm: null,
    rightArm: null,
  });

  // Load the MoveNet model
  const loadModel = async () => {
    await tf.setBackend("webgl");
    await tf.ready();
    const model = await loadMoveNet("singlepose/lightning");
    setModel(model);
    console.log("MoveNet Model Loaded");
  };

  // Function to detect the pose
  const detectPose = async () => {
    if (model && videoRef.current) {
      const poses = await model.estimateSinglePose(videoRef.current, {
        flipHorizontal: false,
      });

      // Extract keypoints
      const keypoints = poses.keypoints;

      // Get canvas context to draw on top of video
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      // Clear the canvas before drawing new keypoints
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw keypoints as circles on the canvas
      keypoints.forEach((point) => {
        if (point.score >= 0.5) {
          const { x, y } = point.position;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "red";
          ctx.fill();
        }
      });

      // Optionally, connect keypoints with lines (e.g., drawing skeleton)
      drawSkeleton(keypoints, ctx);

      // Example: Calculate the angle between the shoulder, elbow, and wrist
      const leftShoulder = keypoints.find((point) => point.part === "leftShoulder");
      const leftElbow = keypoints.find((point) => point.part === "leftElbow");
      const leftWrist = keypoints.find((point) => point.part === "leftWrist");

      const rightShoulder = keypoints.find((point) => point.part === "rightShoulder");
      const rightElbow = keypoints.find((point) => point.part === "rightElbow");
      const rightWrist = keypoints.find((point) => point.part === "rightWrist");

      if (leftShoulder && leftElbow && leftWrist) {
        const leftArmAngle = calculateAngle(
          leftShoulder.position,
          leftElbow.position,
          leftWrist.position
        );
        setAngles((prev) => ({ ...prev, leftArm: leftArmAngle }));
      }

      if (rightShoulder && rightElbow && rightWrist) {
        const rightArmAngle = calculateAngle(
          rightShoulder.position,
          rightElbow.position,
          rightWrist.position
        );
        setAngles((prev) => ({ ...prev, rightArm: rightArmAngle }));
      }
    }

    requestAnimationFrame(detectPose);
  };

  // Function to draw skeleton (connecting keypoints with lines)
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

  useEffect(() => {
    loadModel();
  }, []);

  const videoStyles = {
    width: "640px",
    height: "480px",
    border: "2px solid black",
  };

  const canvasStyles = {
    position: "absolute",
    top: 0,
    left: 0,
    width: "640px",
    height: "480px",
  };

  return (
    <div style={{ position: "relative" }}>
      <h1>MoveNet Pose Detection</h1>
      <video
        ref={videoRef}
        style={videoStyles}
        controls
        onLoadedMetadata={() => videoRef.current.play()}
      >
        <source src="/path/to/your/video.mp4" type="video/mp4" />
      </video>
      <canvas ref={canvasRef} style={canvasStyles}></canvas>

      {/* Display the calculated angles */}
      {angles.leftArm && (
        <div>
          <h3>Left Arm Angle: {angles.leftArm.toFixed(2)}°</h3>
        </div>
      )}
      {angles.rightArm && (
        <div>
          <h3>Right Arm Angle: {angles.rightArm.toFixed(2)}°</h3>
        </div>
      )}
    </div>
  );
};

export default MoveNet;
