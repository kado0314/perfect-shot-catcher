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
    
    # 非常に強いぼかしを適用 (背景用)
    max_blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # 出力画像を準備
    output_img = np.copy(img).astype(np.float32)

    # 3. ぼかしのグラデーションマスクを作成し、ブレンド
    
    # グラデーションが始まる領域を設定 (ピント領域の端から外側へ)
    fade_start_y_upper = center_y - height_y // 2
    fade_end_y_upper = max(0, fade_start_y_upper - h // 4) # 上方向のフェードが完了する位置
    
    fade_start_y_lower = center_y + height_y // 2
    fade_end_y_lower = min(h, fade_start_y_lower + h // 4) # 下方向のフェードが完了する位置

    for y in range(h):
        alpha = 0.0 # 0: オリジナル (シャープ), 1.0: 最大ぼかし
        
        # 上部グラデーションの計算
        if y < fade_start_y_upper:
            if y < fade_end_y_upper:
                alpha = 1.0 # 最大ぼかし
            else:
                # 線形グラデーション
                range_y = fade_start_y_upper - fade_end_y_upper
                dist_y = fade_start_y_upper - y
                alpha = dist_y / range_y if range_y > 0 else 0.0

        # 下部グラデーションの計算
        elif y > fade_start_y_lower:
            if y > fade_end_y_lower:
                alpha = 1.0 # 最大ぼかし
            else:
                # 線形グラデーション
                range_y = fade_end_y_lower - fade_start_y_lower
                dist_y = y - fade_start_y_lower
                alpha = dist_y / range_y if range_y > 0 else 0.0
        
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
st.set_page_config(layout="wide", page_title="ミニチュア効果メーカー (Tilt-Shift)")

st.title("🏡 ミニチュア効果メーカー (Tilt-Shift)")
st.caption("風景写真をアップロードすると、最適なデフォルト設定で自動的にジオラマ風効果が適用されます。")

# 1. ファイルアップローダー
uploaded_file = st.file_uploader("風景画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 2. 画像の読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)

    # 3. 設定サイドバー
    with st.sidebar:
        st.header("⚙️ 効果設定")
        st.markdown("以下の設定を変更すると、結果が**自動で**更新されます。")
        
        # 自動設定のための最適化されたデフォルト値を設定
        focus_center = st.slider("ピントの中心 (縦方向)", 0.0, 1.0, 0.55, 0.01) 
        focus_height = st.slider("ピントの幅 (縦方向)", 0.0, 0.5, 0.2, 0.01)
        blur_strength = st.slider("最大ぼかしの強さ (カーネルサイズ)", 5, 101, 71, step=2)
        saturation_boost = st.slider("彩度の強調", 0, 100, 40)
        
    # 4. 結果の表示と自動実行
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("オリジナル画像")
        st.image(original_image, channels="BGR", use_column_width=True)

    # Streamlitはスライダーが変更されるたびにこのブロックを再実行します
    try:
        with st.spinner('画像を自動処理中... (初回や強力なぼかしは時間がかかることがあります)'): 
            processed_image = apply_tilt_shift(
                original_image, 
                focus_center, 
                focus_height, 
                blur_strength, 
                saturation_boost
            )

            with col2:
                st.subheader("ミニチュア効果 (自動適用)")
                st.image(processed_image, channels="BGR", use_column_width=True)
                
                # ダウンロードボタン
                _, encoded_img = cv2.imencode('.png', processed_image)
                st.download_button(
                    label="結果画像をダウンロード (PNG)",
                    data=encoded_img.tobytes(),
                    file_name="miniature_output.png",
                    mime="image/png"
                )
    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
        st.info("カーネルサイズが大きすぎるか、画像サイズが小さすぎることが原因かもしれません。設定を調整してください。")
        
