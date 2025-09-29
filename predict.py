import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple


def predict_single_image(
    image_path: str,
    models: List[torch.nn.Module],
    class_map: Dict[int, str],
    tta_transforms,
    num_tta: int = 5,
    device: torch.device = None,
) -> Tuple[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if img.shape[0] > img.shape[1]:
        img = np.rot90(img)

    h, w, _ = img.shape
    if h != w:
        size = max(h, w)
        pad_top = (size - h) // 2
        pad_bottom = size - h - pad_top
        pad_left = (size - w) // 2
        pad_right = size - w - pad_left
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    original_padded_image = img.copy()

    for m in models:
        m.to(device)
        m.eval()

    all_tta_preds = []

    for _ in range(num_tta):
        if tta_transforms:
            augmented = tta_transforms(image=original_padded_image)
            tta_image = augmented["image"]
        else:
            tta_image = original_padded_image.astype(np.float32)

        if isinstance(tta_image, np.ndarray):
            tta_tensor = (
                torch.from_numpy(tta_image.transpose(2, 0, 1))
                .float()
                .unsqueeze(0)
                .to(device)
            )
        else:
            tta_tensor = tta_image.unsqueeze(0).to(device).float()

        ensemble_preds_for_this_tta = []

        with torch.no_grad():
            for model in models:
                outputs = model(tta_tensor)
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1)
                ensemble_preds_for_this_tta.append(probs.float().cpu())

        avg_ensemble_pred = torch.stack(ensemble_preds_for_this_tta, dim=0).mean(dim=0)
        all_tta_preds.append(avg_ensemble_pred)

    tta_stack = torch.stack(all_tta_preds, dim=0)  # (num_tta, 1, num_classes)
    avg_tta = tta_stack.mean(dim=0)  # (1, num_classes)
    final_probs = avg_tta.squeeze(0)  # (num_classes,)

    pred_idx = int(torch.argmax(final_probs).item())
    pred_name = class_map.get(pred_idx, str(pred_idx))
    confidence = float(final_probs[pred_idx].item())

    return pred_name, confidence
