# イラスト領域検出ロジック

このドキュメントでは、`shadowbox.detection.region.RegionDetector` で実装されている10種類のイラスト領域検出ロジックについて説明します。

## 概要

TCGカード画像からイラスト領域を自動検出するために、複数のアプローチを実装しています。各手法には得意・不得意があるため、すべての手法を実行し、**信頼度 × 面積比**のスコアが最も高い結果を採用します。

---

## 検出手法一覧

| メソッド名 | 内部メソッド | 主なアプローチ |
|-----------|-------------|---------------|
| `edge_detection` | `_detect_by_edges` | Cannyエッジ + 輪郭検出 |
| `hsv_threshold` | `_detect_by_hsv_threshold` | HSV色空間での大津閾値処理 |
| `contour_detection` | `_detect_by_contours` | 適応的閾値 + 4角形近似 |
| `grid_scoring` | `_detect_by_grid_scoring` | グリッド分割 + スライディングウィンドウ |
| `boundary_contrast` | `_detect_by_boundary_contrast` | 内部複雑度 vs 境界単色度 |
| `frame_analysis` | `_detect_by_frame_analysis` | フレーム色から内側へ探索 |
| `band_complexity` | `_detect_by_band_complexity` | 水平帯の複雑度分析 |
| `horizontal_lines` | `_detect_by_horizontal_lines` | Hough変換で水平線検出 |
| `center_expansion` | `_detect_by_center_expansion` | 中央から複雑度で拡張 |
| `gradient_richness` | `_detect_by_gradient_richness` | 勾配の豊かさで境界検出 |

---

## 各手法の詳細

### 1. edge_detection（エッジ検出）

**アルゴリズム:**
1. グレースケール変換
2. Cannyエッジ検出（低閾値/高閾値で二重閾値処理）
3. 輪郭検出（`cv2.findContours`）
4. 面積が適切な範囲の矩形輪郭を抽出

**特徴:**
- シンプルで高速
- コントラストが明確な画像に有効
- ノイズが多い画像では誤検出しやすい

**パラメータ:**
- `canny_low`: Cannyの低閾値（デフォルト: 50）
- `canny_high`: Cannyの高閾値（デフォルト: 150）

---

### 2. hsv_threshold（HSV閾値処理）

**アルゴリズム:**
1. RGB→HSV色空間に変換
2. 彩度（S）チャンネルに大津の二値化を適用
3. 明度（V）チャンネルにも大津の二値化を適用
4. 両方のマスクを組み合わせて領域を特定

**特徴:**
- 色彩豊かなイラストと単色のフレームを区別しやすい
- 白黒やセピア調のイラストには不向き

**使用技術:**
- `cv2.threshold` with `THRESH_OTSU`

---

### 3. contour_detection（輪郭検出）

**アルゴリズム:**
1. グレースケール変換
2. 適応的閾値処理（`cv2.adaptiveThreshold`）
3. 輪郭検出
4. 4角形に近似できる輪郭を探索（`cv2.approxPolyDP`）

**特徴:**
- 矩形の枠線がはっきりしている場合に有効
- 複雑な形状のイラスト枠には不向き

---

### 4. grid_scoring（グリッドスコアリング）

**アルゴリズム:**
1. 画像をN×Mのグリッドに分割
2. 各セルの「イラストらしさ」スコアを計算:
   - 彩度の分散
   - エッジ密度
   - 色相の多様性
3. スライディングウィンドウで最もスコアの高い矩形領域を探索

**特徴:**
- グローバルな視点で領域を評価
- 計算コストがやや高い
- 信頼度が高くなりやすい（1.0になることも）

**スコア計算式:**
```
score = 彩度分散 × エッジ密度 × 色相多様性
```

---

### 5. boundary_contrast（境界コントラスト）

**アルゴリズム:**
1. 候補領域の内部の複雑度を計算
2. 候補領域の境界（外側の帯）の単色度を計算
3. 「内部が複雑で、境界が単純」な領域を高スコアとする

**特徴:**
- フレームが単色の場合に有効
- イラストがフレームまではみ出している場合は精度低下

**信頼度計算:**
```
confidence = (内部複雑度) × (1 - 境界複雑度)
```

---

### 6. frame_analysis（フレーム解析）

**アルゴリズム:**
1. 画像の四辺（上下左右）からフレーム色をサンプリング
2. 各辺から内側に向かって探索
3. フレーム色から大きく変化する位置を境界とする

**特徴:**
- 外側から内側へ探索するアプローチ
- フレーム色が一定の場合に有効
- グラデーションのあるフレームには不向き

---

### 7. band_complexity（水平帯複雑度）

**アルゴリズム:**
1. 画像を水平方向にN分割（帯）
2. 各帯の複雑度を計算:
   - 彩度の分散
   - エッジ密度
   - 色相の多様性
3. 複雑度が高い連続した帯をイラスト領域とする

**特徴:**
- TCGカードの構造（タイトル→イラスト→テキスト→パワー）に適合
- 垂直方向の境界検出には別の手法が必要

**帯の複雑度:**
```
complexity = saturation_var × edge_density × hue_diversity
```

---

### 8. horizontal_lines（水平線検出）

**アルゴリズム:**
1. Cannyエッジ検出
2. Hough変換で直線を検出（`cv2.HoughLinesP`）
3. 水平に近い線（±10度以内）を抽出
4. 上部の区切り線（タイトル下）と下部の区切り線を特定

**特徴:**
- カードの区切り線が明確な場合に非常に有効
- 区切り線がない/曖昧な場合は検出失敗

**パラメータ:**
- `minLineLength`: 画像幅の50%以上
- `maxLineGap`: 10ピクセル
- 角度閾値: ±10度

---

### 9. center_expansion（中央拡張）

**アルゴリズム:**
1. 画像の中央領域（40%×40%）の複雑度を基準とする
2. 中央から上下左右に拡張
3. 複雑度が基準の30%以下に下がる位置を境界とする

**特徴:**
- イラストが中央にある前提のアプローチ
- 連続チェック（3ピクセル連続で閾値以下）でノイズに強い

**複雑度計算（各ライン）:**
```
complexity = saturation_std × edge_ratio × (hue_unique / 36)
```

---

### 10. gradient_richness（勾配の豊かさ）⭐推奨

**アルゴリズム:**
1. グレースケール変換
2. Sobelフィルタで勾配（微分）を計算
   - `grad_x`: 水平方向の勾配
   - `grad_y`: 垂直方向の勾配
   - `magnitude = √(grad_x² + grad_y²)`
3. 各垂直/水平ラインの勾配の豊かさ（分散）を計算
4. 中央領域の平均を基準に、25%以下に下がる位置を境界とする

**特徴:**
- 色情報を使わずテクスチャの豊かさで判定
- タイトルのテキストエッジと区別しやすい
- 多くのTCGカードで良好な結果

**数式:**
```python
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# 各ラインの豊かさ = 勾配magnitude の分散
richness = np.var(magnitude[line])

# 閾値 = 中央領域の平均 × 0.25
threshold = center_avg * 0.25
```

**なぜ有効か:**
- イラスト領域: 様々な強度の勾配が混在 → 分散が大きい
- フレーム/テキスト領域: 均一または規則的 → 分散が小さい

---

## スコアリングとランキング

すべての検出手法の結果を以下のスコアでランキングします:

```python
score = confidence × (bbox_area / image_area)
```

- **confidence**: 各手法が出力する信頼度（0.0〜1.0）
- **bbox_area**: 検出された矩形の面積
- **image_area**: 画像全体の面積

面積比を掛けることで、適切なサイズの領域を優先します。

---

## 使用方法

### 自動検出（最良の結果を取得）

```python
from shadowbox.detection import detect_illustration_region

result = detect_illustration_region(image)
print(f"Method: {result.method}, Confidence: {result.confidence}")
print(f"BBox: {result.bbox}")
```

### 特定の手法を直接使用

```python
from shadowbox.detection import RegionDetector

detector = RegionDetector()
cv_image = detector._pil_to_cv(image)

# gradient_richness を直接使用
result = detector._detect_by_gradient_richness(cv_image)
```

### 全候補を取得

```python
detector = RegionDetector()
candidates = detector.detect_with_candidates(image, max_candidates=10)

for r in candidates:
    print(f"{r.method}: {r.confidence:.2f}")
```

---

## パラメータ調整

`RegionDetector` のコンストラクタで以下のパラメータを調整できます:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `min_area_ratio` | 0.1 | 検出領域の最小面積比（画像全体に対する） |
| `max_area_ratio` | 0.9 | 検出領域の最大面積比 |
| `canny_low` | 50 | Cannyエッジ検出の低閾値 |
| `canny_high` | 150 | Cannyエッジ検出の高閾値 |

```python
detector = RegionDetector(
    min_area_ratio=0.15,
    max_area_ratio=0.85,
    canny_low=30,
    canny_high=100,
)
```
