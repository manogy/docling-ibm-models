# IOCR and Docling Models - Disabled Models Analysis

## Summary

Based on analysis of the IOCR configuration files and code, here are the models that are **DISABLED** or **NOT USED** in the current WDU setup:

---

## 1. Script Recognition Model (DISABLED)

**Status:** ❌ DISABLED

**Configuration:** [`../IOCR/hrl_ocr/pipeline/ibm_ocr/app_config/default_iocr_init_config.json:189`](../IOCR/hrl_ocr/pipeline/ibm_ocr/app_config/default_iocr_init_config.json:189)
```json
"script_recognition" : {
    "is_enabled": false,
    ...
}
```

**Code Check:** [`../IOCR/hrl_ocr/pipeline/ibm_ocr/ocr_inference.py:718`](../IOCR/hrl_ocr/pipeline/ibm_ocr/ocr_inference.py:718)
```python
def __load_script_recognition_model(self, script_recognition_config, recognition_config, script_mapping_config):
    if not script_recognition_config['is_enabled']:
        return None  # Model not loaded
```

**Purpose:** Script classification model for identifying different writing scripts (Greek, Hebrew, Cyrillic, Arabic, Latin, etc.)

**Why Disabled:** Not currently needed in production workflow

---

## 2. TrOCR Handwriting Recognition Model (NOT USED)

**Status:** ❌ NOT USED (Alternative model type available but not selected)

**Configuration:** [`../IOCR/hrl_ocr/pipeline/ibm_ocr/app_config/default_iocr_init_config.json:252-256`](../IOCR/hrl_ocr/pipeline/ibm_ocr/app_config/default_iocr_init_config.json:252)

```json
"eng_hw" : {
    "model_type": "ssocr",  // ✅ USING THIS
    "comment_model_types": ["ssocr", "trocr"],  // trocr available but not used
    "use_cuda": true,
    "model_path" : "./models/recognition/ai_hwr/weights/aihwr_checkpoint-39606",  // TrOCR model path
    ...
    "weights" : "./models/recognition/ssocr/weights/eng_hw/model_last_eng_hw.pth",  // ✅ SSOCR weights being used
}
```

**Details:**
- **Two model types available:** `ssocr` and `trocr`
- **Currently using:** `ssocr` (already in ONNX format)
- **Not using:** `trocr` (ai_hwr model)
- **SSOCR model location:** `recognition/ssocr/weights/eng_hw/` (already ONNX)
- **TrOCR model location:** `recognition/ai_hwr/weights/` (NOT being used)

**Why Not Used:** SSOCR provides sufficient accuracy and is already optimized in ONNX format

---

## 3. German Handwriting (deu_hw) - Same Pattern

**Status:** ❌ TrOCR variant NOT USED

Similar to `eng_hw`, German handwriting also has:
- ✅ **Using:** SSOCR model (`recognition/ssocr/weights/deu_hw/`)
- ❌ **Not using:** TrOCR variant

---

## Models That ARE Being Used (For Reference)

### IOCR Models (ENABLED):
1. **UNet Detection** - Word detection model
2. **TDMv2** - Text detection model v2
3. **TDMv2_2** - Text detection model v2.2
4. **SSOCR Recognition** - Text recognition for multiple languages (eng, deu, fra, spa, ita, por, nld, etc.)
5. **SSOCR Handwriting** - Handwriting recognition (eng_hw, deu_hw)
6. **Language Detection** - Language identification

### Docling Models (ENABLED):
1. **Layout Model** - Page layout analysis
2. **Table Model** - Table structure recognition
3. **Figure Classifier** - Figure/chart classification
4. **Code/Formula Model** - Code and formula detection

---

## ZDLC Conversion Requirements

### Models That Need ZDLC Conversion:
✅ **Docling Models** (PyTorch → ONNX → ZDLC):
- Layout predictor
- Table predictor (TableFormer)
- Figure classifier
- Code/Formula predictor

### Models Already in ONNX (No Conversion Needed):
✅ **IOCR SSOCR Models** (already ONNX):
- All SSOCR recognition models
- SSOCR handwriting models (eng_hw, deu_hw)

### Models NOT Needed for ZDLC:
❌ **Disabled/Unused Models**:
- Script recognition model (disabled)
- TrOCR/ai_hwr models (not being used)

---

## Recommendations

1. **No need to convert disabled models** - Script recognition is disabled, so no ZDLC conversion needed
2. **No need to convert TrOCR** - The ai_hwr (TrOCR) model is not being used; SSOCR is the active model
3. **Focus ZDLC efforts on:**
   - Docling models (layout, table, figure, code/formula)
   - These are the PyTorch models that need ONNX → ZDLC conversion
4. **IOCR SSOCR models** - Already in ONNX format, can potentially be converted to ZDLC if needed for IBM Z optimization

---

## Configuration Files Analyzed

1. [`../IOCR/hrl_ocr/pipeline/ibm_ocr/app_config/default_iocr_init_config.json`](../IOCR/hrl_ocr/pipeline/ibm_ocr/app_config/default_iocr_init_config.json)
2. [`../IOCR/hrl_ocr/pipeline/ibm_ocr/ocr_inference.py`](../IOCR/hrl_ocr/pipeline/ibm_ocr/ocr_inference.py)
3. [`../docling-ibm-models/docling_ibm_models/`](../docling-ibm-models/docling_ibm_models/)

---

## Updated Architecture Notes

The WDU architecture diagram should reflect:
- **Script Recognition Model:** DISABLED (not loaded)
- **Handwriting Recognition:** Uses SSOCR only (TrOCR/ai_hwr not used)
- **All other IOCR models:** ENABLED and active
- **All Docling models:** ENABLED and active