from flask import Flask, request, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

@app.route('/')
def index():
    return send_file('video.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video1 = request.files['video1']
    video2 = request.files['video2']

    # Save the uploaded video files
    video1_path = 'test.mp4.mov'
    video2_path = 'test2.mp4'
    video1.save(video1_path)
    video2.save(video2_path)

    # Analyze the baseball swings
    angles1, hip_angles1, shoulder_angles1 = analyze_baseball_swing(video1_path, 'output1.mp4')
    angles2, hip_angles2, shoulder_angles2 = analyze_baseball_swing(video2_path, 'output2.mp4')

    # Calculate similarity percentages
    min_length = min(len(angles1), len(angles2))
    angles1 = angles1[:min_length]
    angles2 = angles2[:min_length]

    angle_diffs = [abs(a1 - a2) for a1, a2 in zip(angles1, angles2)]
    avg_angle_diff = sum(angle_diffs) / len(angle_diffs)

    similarity_percentage = 100 - (avg_angle_diff / 180) * 100

    min_length = min(len(hip_angles1), len(hip_angles2))
    hip_angles1 = hip_angles1[:min_length]
    hip_angles2 = hip_angles2[:min_length]

    hip_angle_diffs = [abs(a1 - a2) for a1, a2 in zip(hip_angles1, hip_angles2)]
    avg_hip_angle_diff = sum(hip_angle_diffs) / len(hip_angle_diffs)

    hip_similarity_percentage = 100 - (avg_hip_angle_diff / 180) * 100

    min_length = min(len(shoulder_angles1), len(shoulder_angles2))
    shoulder_angles1 = shoulder_angles1[:min_length]
    shoulder_angles2 = shoulder_angles2[:min_length]

    shoulder_angle_diffs = [abs(a1 - a2) for a1, a2 in zip(shoulder_angles1, shoulder_angles2)]
    avg_shoulder_angle_diff = sum(shoulder_angle_diffs) / len(shoulder_angle_diffs)

    shoulder_similarity_percentage = 100 - (avg_shoulder_angle_diff / 180) * 100

    # Prepare the response data
    response_data = {
        'left_arm_angle_similarity': similarity_percentage,
        'hip_angle_similarity': hip_similarity_percentage,
        'shoulder_angle_similarity': shoulder_similarity_percentage
    }

    return jsonify(response_data)

def analyze_baseball_swing(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    angles = []
    hip_angles = []
    shoulder_angles = []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Convert the image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                # Calculate angles
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                hip_angle = calculate_angle(left_shoulder, left_hip, right_hip)
                shoulder_angle = calculate_angle(left_elbow, left_shoulder, right_shoulder)

                angles.append(left_arm_angle)
                hip_angles.append(hip_angle)
                shoulder_angles.append(shoulder_angle)

                # Visualize angles
                cv2.putText(image, str(left_arm_angle), 
                            tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(hip_angle), 
                            tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(shoulder_angle), 
                            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Write the processed frame to the output video
            out.write(image)

    cap.release()
    out.release()
    
    return angles, hip_angles, shoulder_angles

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

if __name__ == '__main__':
    app.run()
