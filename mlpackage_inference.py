import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import coremltools as ct
from transformers import AutoImageProcessor, AutoConfig

# ---------------------------------------------------------------------------
# 1. 설정 & 로드
# ---------------------------------------------------------------------------
mlmodel_path = "EOMT_2.mlpackage" 
model_id = "tue-mps/coco_panoptic_eomt_small_640_2x"

# 테스트할 이미지 인덱스 설정
image_names= ["000000015497", "000000104572", "000000130699", "000000131273", "000000161861", "000000261116", "000000356424", "000000377393", "000000389315", "000000391648"]
idx = 9
image_path = f'evaluation/val_images_random10/images/{image_names[idx]}.jpg'
#image_path = f'evaluation/val_images_random10/images/Marketplace.jpg'
print(f"📥 Loading Processor & Config...")
processor = AutoImageProcessor.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
id2label = config.id2label 

print(f"🚀 Loading Core ML model...")
model = ct.models.MLModel(mlmodel_path)

# ---------------------------------------------------------------------------
# 2. 입력 데이터 준비 (Letterbox Resize)
# ---------------------------------------------------------------------------
image = Image.open(image_path).convert("RGB")

def resize_with_padding(image, target_size=(640, 640)):
    target_w, target_h = target_size
    orig_w, orig_h = image.size
    
    # 비율 유지 리사이즈 비율 계산
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    
    resized_image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # 검은 배경 생성
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image, (paste_x, paste_y, new_w, new_h)

# 이미지 리사이즈 및 패딩 정보 저장
input_image, pad_info = resize_with_padding(image, target_size=(640, 640))
paste_x, paste_y, new_w, new_h = pad_info  # 나중에 자를 때 사용

# ---------------------------------------------------------------------------
# 3. 추론 (Core ML)
# ---------------------------------------------------------------------------
print("🔮 Running Core ML Prediction...")
preds = model.predict({"pixel_values": input_image})

# ---------------------------------------------------------------------------
# 4. 후처리 (Post-processing) - 데이터 타입 수정됨
# ---------------------------------------------------------------------------
class CoreMLOutputWrapper:
    def __init__(self, class_logits, mask_logits):
        self.class_queries_logits = torch.from_numpy(class_logits)
        self.masks_queries_logits = torch.from_numpy(mask_logits)

c_logits = preds["class_logits"]
m_logits = preds["mask_logits"]

if c_logits.ndim == 2: c_logits = np.expand_dims(c_logits, 0)
if m_logits.ndim == 3: m_logits = np.expand_dims(m_logits, 0)

outputs = CoreMLOutputWrapper(c_logits, m_logits)

print("⚙️ Post-processing...")

# [1단계] 일단 패딩이 포함된 640x640 크기로 결과를 받습니다.
final_preds = processor.post_process_panoptic_segmentation(
    outputs,
    target_sizes=[(640, 640)], 
    threshold=0.8
)

# 640x640 결과 추출
seg_640 = final_preds[0]["segmentation"].cpu().numpy()
segments_info = final_preds[0]["segments_info"]

# [2단계] 패딩 제거 (Crop)
seg_cropped = seg_640[paste_y : paste_y + new_h, paste_x : paste_x + new_w]

# [3단계] 원본 크기로 복원 (Resize)
# ⚠️ [수정된 부분] int64 -> int32 변환 추가 (PIL 호환성 문제 해결)
seg_cropped = seg_cropped.astype(np.int32) 

seg_pil = Image.fromarray(seg_cropped)
seg_resized = seg_pil.resize(image.size, resample=Image.NEAREST)
seg_final = np.array(seg_resized)

# ---------------------------------------------------------------------------
# [수정] 시각화 (오버레이 + 라벨 텍스트) - 고정 색상 로직 적용
# ---------------------------------------------------------------------------
print(f"Found {len(segments_info)} segments.")
H, W = seg_final.shape 

# 빈 도화지 생성
color_img = np.zeros((H, W, 3), dtype=np.uint8)

# ✨ [핵심 변경] ID 기반 색상 생성 함수
def get_stable_color(id_value):
    # ID 값을 시드로 사용하여 항상 똑같은 랜덤 색을 만듦
    rng_stable = np.random.default_rng(id_value)
    return rng_stable.integers(0, 255, size=3, dtype=np.uint8)

for s in segments_info:
    segment_id = s["id"]
    label_id = s["label_id"]
    
    # 방법 A: 세그먼트 ID 기준 (같은 객체는 항상 같은 색) - 추천
    color = get_stable_color(segment_id)
    
    # 방법 B: 클래스 기준 (모든 '사람'은 같은 색) - 원하면 이걸로 교체
    # color = get_stable_color(label_id)

    # 색칠하기
    color_img[seg_final == segment_id] = color

# (이하 오버레이 및 텍스트 코드는 동일)
overlay = Image.blend(image.convert("RGBA"), Image.fromarray(color_img).convert("RGBA"), alpha=0.6)
# 2) 플롯 그리기
plt.figure(figsize=(12, 12))
plt.imshow(overlay)
plt.axis("off")

# 3) 라벨 텍스트 찍기 루프
print("🏷 Adding labels...")
for s in segments_info:
    segment_id = s["id"]
    label_id = s["label_id"]
    score = s.get("score", None)
    
    # 원본 크기 마스크 기준으로 좌표 찾기
    mask = (seg_final == segment_id)
    ys, xs = np.where(mask) 
    
    if len(ys) == 0:
        continue
        
    # 무게 중심(Center of Mass) 계산
    cy, cx = np.mean(ys), np.mean(xs)
    
    # 라벨 이름 가져오기
    label_name = id2label.get(label_id, str(label_id))
    
    # 텍스트 구성
    txt = f"{label_name}"
    if score is not None:
        txt += f"\n{score:.2f}"
    
    # 텍스트 그리기
    plt.text(
        cx, cy, txt, 
        color="white", 
        fontsize=9, 
        fontweight='bold',
        ha="center", va="center",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
    )

plt.title(f"Core ML Panoptic Segmentation ({len(segments_info)} objects)")
plt.tight_layout()
plt.show()