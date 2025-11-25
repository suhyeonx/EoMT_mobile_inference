import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from transformers import EomtForUniversalSegmentation, AutoImageProcessor

# ---------------------------------------------------------------------------
# 1. 모델 및 프로세서 로드
# ---------------------------------------------------------------------------
model_id = "tue-mps/coco_panoptic_eomt_small_640_2x"
print(f"📥 Loading model & processor: {model_id}...")

# Processor에서 설정값 가져오기
processor = AutoImageProcessor.from_pretrained(model_id)
base_model = EomtForUniversalSegmentation.from_pretrained(model_id)
base_model.eval()

# ---------------------------------------------------------------------------
# 2. [핵심] ImageNet Mean/Std 역산하여 Core ML 파라미터 계산
# ---------------------------------------------------------------------------
# PyTorch 공식: output = (image/255.0 - mean) / std
# Core ML 공식: output = (image * scale) + bias
# 따라서:
# scale = 1 / (255.0 * std)
# bias  = -mean / std

image_mean = np.array(processor.image_mean) # [0.485, 0.456, 0.406]
image_std = np.array(processor.image_std)   # [0.229, 0.224, 0.225]

print(f"📊 Processor Mean: {image_mean}")
print(f"📊 Processor Std : {image_std}")

# Core ML ImageType의 scale은 단일 float 값만 허용되는 경우가 많음 (채널별 차이가 크지 않으므로 평균 사용)
# 미세한 차이를 줄이기 위해 bias 계산 시에는 각 채널별 std를 반영
avg_std = np.mean(image_std) 

scale = 1.0 / (255.0 * avg_std)
bias = (-image_mean / image_std).tolist() # RGB 채널별 Bias

print(f"🧮 Calculated Scale: {scale}")
print(f"🧮 Calculated Bias : {bias}")

# ---------------------------------------------------------------------------
# 3. Wrapper (Dict -> Tuple)
# ---------------------------------------------------------------------------
class EomtWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # x는 Core ML이 전처리한 Tensor가 들어옴
        outputs = self.model(pixel_values=x)
        return outputs.class_queries_logits, outputs.masks_queries_logits

wrapper_model = EomtWrapper(base_model)

# ---------------------------------------------------------------------------
# 4. Tracing
# ---------------------------------------------------------------------------
# Trace용 더미 입력 (값은 상관없음, 형태만 중요)
dummy_input = torch.rand(1, 3, 640, 640)
print("🎥 Tracing model...")
traced_model = torch.jit.trace(wrapper_model, dummy_input)

# ---------------------------------------------------------------------------
# 5. Core ML 변환 (정확도 최우선 설정)
# ---------------------------------------------------------------------------
print("📦 Converting to Core ML Package...")

model_ct = ct.convert(
    traced_model,
    inputs=[
        ct.ImageType(
            name="pixel_values", 
            shape=(1, 3, 640, 640), # 모델의 학습 해상도
            scale=scale, 
            bias=bias,
            color_layout=ct.colorlayout.RGB # 명시적 RGB 지정
        )
    ],
    outputs=[
        ct.TensorType(name="class_logits"),
        ct.TensorType(name="mask_logits")
    ],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
    
    # 👇 [가장 중요] 결과가 안 좋은 결정적 원인 해결 (FP32 강제)
    compute_precision=ct.precision.FLOAT32
)

# 메타데이터 추가 (선택사항)
model_ct.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
model_ct.short_description = "EOMT Panoptic Segmentation (FP32)"

save_path = "EOMT_2.mlpackage"
model_ct.save(save_path)
print(f"✅ Success! Saved '{save_path}' with FLOAT32 precision.")