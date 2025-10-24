import cv2
import numpy as np

def apply_tilt_shift(image_path, focus_center_y, focus_height, max_blur_kernel=51):
    # 1. 画像の読み込み
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    h, w, _ = img.shape
    
    # 2. ぼかしカーネルの準備
    # ティルトシフトは「ぼかしのグラデーション」が重要
    
    # 3. ぼかしの適用（例: グラデーションマスクとガウシアンブラーを使用）
    output_img = np.copy(img)
    
    # ピントを合わせたい領域 (focus_center_yを中心としたfocus_heightの範囲)
    start_y = max(0, int(focus_center_y - focus_height / 2))
    end_y = min(h, int(focus_center_y + focus_height / 2))
    
    # --- ぼかし処理のロジック ---
    # ここが少し複雑になりますが、複数のぼかしレベルの画像をマスクで合成します
    # 例: 
    #   - 弱ぼかし (center_blur)
    #   - 強ぼかし (max_blur)
    
    # 強ぼかしを適用した画像を作成 (上下端用)
    max_blur_img = cv2.GaussianBlur(img, (max_blur_kernel, max_blur_kernel), 0)
    
    # ぼかしの強さを制御するグラデーションマスクの作成
    mask = np.zeros((h, w), dtype=np.float32)

    # 中心から上下端に向かって0（シャープ）から1（最大ぼかし）へグラデーション
    # ※ このグラデーションマスクの作成が肝になります
    
    # マスクを使ってブレンド
    for y in range(h):
        # ここでy座標に応じたブレンド率(alpha)を計算する
        # alpha = 0 (ピント域) から alpha = 1 (上下端)
        # output_img[y, :] = img[y, :] * (1 - alpha) + max_blur_img[y, :] * alpha
        pass # 実際の計算ロジックを実装
        
    # 4. 彩度・コントラストの調整（ミニチュア感の強調）
    # 例: HSV色空間に変換して彩度(S)を調整
    hsv = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 50) # 彩度を上げる例
    result_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 5. 結果の保存
    cv2.imwrite("output_miniature.jpg", result_img)
    print("ミニチュア効果の画像を output_miniature.jpg として保存しました。")


# 実行例
# focus_center_y: 画像の高さに対するピントの中心 (例: h/2)
# focus_height: ピントの合う範囲の高さ (例: h/5)
# apply_tilt_shift("sample_input.jpg", 250, 100)
