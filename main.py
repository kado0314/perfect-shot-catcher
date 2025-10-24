import cv2
import numpy as np
import dlib
import time
import os

# --- 設定値 ---
# Dlibの学習済みモデルファイル（事前にダウンロードし、プロジェクトルートに配置）
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
# 目の縦横比の閾値 (EAR: Eye Aspect Ratio)。これ以下だと目を閉じていると判断
EAR_THRESHOLD = 0.25 
# 連続撮影時間（トリガー後）
BURST_DURATION_SEC = 2.0 
# プリトリガー（バッファ）の時間（トリガー前の何秒を保存するか）
PRE_TRIGGER_SEC = 1.0 
# フレームレート (FPS)
FPS = 10 

# --- Dlibの初期化 ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# 目ランドマークのインデックス（Dlib 68点モデルの場合）
# 左目: 36-41, 右目: 42-47
L_START, L_END = 36, 42
R_START, R_END = 42, 48

def eye_aspect_ratio(eye):
    # 目の縦の距離を計算 ((P2-P6) + (P3-P5))
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # 目の横の距離を計算 (P1-P4)
    C = np.linalg.norm(eye[0] - eye[3])
    # EARを計算
    return (A + B) / (2.0 * C)

def start_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けません。カメラが接続されているか、権限を確認してください。")
        return

    # バッファリング用のリスト（プリトリガー用）
    frame_buffer = [] 
    is_bursting = False
    burst_start_time = 0

    print("カメラ起動中... 全員の目が開いた瞬間に高速連写します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        
        all_eyes_open = True
        num_faces = len(faces)
        
        # --- 1. 顔と目の検出とEAR計算 ---
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks_np = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

            # 目の座標を抽出
            left_eye = landmarks_np[L_START:L_END]
            right_eye = landmarks_np[R_START:R_END]

            # EARを計算
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # 目の開閉状態をチェック
            if avg_ear < EAR_THRESHOLD:
                all_eyes_open = False
            
            # 検出結果の描画
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (face.left(), face.top() - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0), 2)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        # --- 2. プリトリガー・バッファリング ---
        # バッファに現在のフレームを追加
        frame_buffer.append(frame.copy())
        # バッファが規定サイズを超えたら、古いフレームを削除
        if len(frame_buffer) > int(PRE_TRIGGER_SEC * FPS):
             frame_buffer.pop(0)

        # --- 3. トリガーチェックとバースト撮影 ---
        if not is_bursting:
            # バースト中でない場合、トリガー条件をチェック
            if num_faces > 0 and all_eyes_open:
                print("トリガー成立！バースト撮影開始！")
                is_bursting = True
                burst_start_time = time.time()
                
                # プリトリガー画像を保存
                for i, buffered_frame in enumerate(frame_buffer):
                    cv2.imwrite(f"burst/shot_{time.time()}_{i:02d}_PRE.jpg", buffered_frame)
                
                # バッファをクリア（後のポストトリガー用）
                frame_buffer.clear()
        
        if is_bursting:
            # ポストトリガー撮影
            elapsed = time.time() - burst_start_time
            if elapsed < BURST_DURATION_SEC:
                # ポストトリガー画像を保存
                cv2.imwrite(f"burst/shot_{time.time()}_{int(elapsed*FPS):02d}_POST.jpg", frame)
                cv2.putText(frame, "RECORDING...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                is_bursting = False
                print("バースト撮影完了。画像が 'burst' フォルダに保存されました。")

        # --- 4. 画面表示 ---
        cv2.putText(frame, f"Faces: {num_faces}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Perfect Shot Catcher', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    # 保存フォルダの作成
    os.makedirs('burst', exist_ok=True)
    start_camera()
