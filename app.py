import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- ティルトシフト効果の核となる処理関数 ---
def apply_tilt_shift(img_array, focus_center_ratio, focus_height_ratio, max_blur_kernel, saturation_boost):
    """
    画像にミニチュア効果（ティルトシフト）を適用します。
    """
    # OpenCVはBGR形式で画像を扱います
    img = img_array 
    h, w, _ = img.shape
    
    # 1. ピント領域の定義 (比率からピクセル値へ)
    center_y = int(h * focus_center_ratio)
    height_y = int(h * focus_height_ratio)
    
    # 2. 最大ぼかし画像の作成
    # ぼかしカーネルは奇数である必要があります
    kernel_size = max_blur_kernel if max_blur_kernel % 2 != 0 else max_blur_kernel + 1
    max_blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    output_img = np.copy(img).astype(np.float32)

    # 3. ぼかしのグラデーションマスクを作成し、ブレンド
    for y in range(h):
        # 中心からの距離を計算 (0.0 から 1.0 の範囲)
        dist_from_center = abs(y - center_y)
        
        # ピントが合うべき領域 (height_y/2) の外側でぼかしを強くする
        dist_outside_focus = max(0, dist_from_center - height_y / 2)
        
        # ぼかしのブレンド率 (alpha): 0 (シャープ) から 1 (最大ぼかし)
        # 距離が増すほど、alphaが1に近づくように計算します。
        # ここでは単純な線形グラデーションを使いますが、s字カーブも効果的です。
        # h/4 をグラデーションの最大距離として仮定
        max_dist_for_fade = h / 4 
        alpha = min(1.0, dist_outside_focus / max_dist_for_fade)
        
        # オリジナル画像と最大ぼかし画像をブレンド
        output_img[y, :] = img[y, :] * (1.0 - alpha) + max_blur_img[y, :] * alpha

    output_img = output_img.astype(np.uint8) # 整数型に戻す

    # 4. 彩度とコントラストの強調（ミニチュア感の強調）
    hsv = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)
    
    # 彩度(S)を調整
    s_channel = hsv[:, :, 1].astype(np.int32)
    s_channel = np.clip(s_channel + saturation_boost, 0, 255).astype(np.uint8)
    hsv[:, :, 1] = s_channel
    
    result_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result_img

# --- Streamlit UI の構築 ---
st.set_page_config(layout="wide", page_title="ミニチュア効果（ティルトシフト）メーカー")

st.title("🏡 ミニチュア効果メーカー (Tilt-Shift)")
st.caption("風景写真をアップロードし、設定を調整して模型のようなジオラマ風効果を適用します。")

# 1. ファイルアップローダー
uploaded_file = st.file_uploader("風景画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 2. 画像の読み込み
    # アップロードされたファイルをNumPy配列（OpenCV形式）に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)

    # 3. 設定サイドバー
    with st.sidebar:
        st.header("⚙️ 効果設定")
        
        # ユーザー入力ウィジェット
        focus_center = st.slider("ピントの中心 (縦方向)", 0.0, 1.0, 0.5, 0.01)
        focus_height = st.slider("ピントの幅 (縦方向)", 0.0, 0.5, 0.2, 0.01)
        blur_strength = st.slider("最大ぼかしの強さ (カーネルサイズ)", 5, 101, 51, step=2)
        saturation_boost = st.slider("彩度の強調", 0, 100, 30)
        
        # 処理実行ボタン
        process_button = st.button("✨ 効果を適用")

    # 4. 結果の表示
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("オリジナル画像")
        st.image(original_image, channels="BGR", use_column_width=True)

    if process_button:
        with st.spinner('画像を処理中... (ぼかし処理に時間がかかることがあります)'):
            # 処理実行
            processed_image = apply_tilt_shift(
                original_image, 
                focus_center, 
                focus_height, 
                blur_strength, 
                saturation_boost
            )

            with col2:
                st.subheader("ミニチュア効果")
                st.image(processed_image, channels="BGR", use_column_width=True)
                
                # ダウンロードボタン
                # OpenCV画像をバイトにエンコード
                _, encoded_img = cv2.imencode('.png', processed_image)
                st.download_button(
                    label="結果画像をダウンロード (PNG)",
                    data=encoded_img.tobytes(),
                    file_name="miniature_output.png",
                    mime="image/png"
                )

    else:
        with col2:
            st.subheader("ミニチュア効果")
            st.info("左の設定を調整し、「効果を適用」ボタンを押してください。")
