import logging
import os
import platform
import threading
from collections.abc import Iterable
from typing import Dict, List, Optional, Set, Union

import numpy as np
from PIL import Image
from transformers import RTDetrImageProcessor

from docling_ibm_models.layoutmodel.labels import LayoutLabels

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()

# Detect architecture at module load time
_IS_S390X = platform.machine().lower() in ['s390x', 's390']

# Conditional imports based on architecture
if _IS_S390X:
    try:
        import zdlc_pyrt
        _ZDLC_AVAILABLE = True
        _log.info("Running on s390x architecture - ZDLC backend will be used")
    except ImportError:
        _ZDLC_AVAILABLE = False
        _log.warning("Running on s390x but zdlc_pyrt not available, falling back to PyTorch")
        import torch
        from torch import Tensor
        from transformers import AutoModelForObjectDetection
else:
    _ZDLC_AVAILABLE = False
    import torch
    from torch import Tensor
    from transformers import AutoModelForObjectDetection
    _log.info(f"Running on {platform.machine()} architecture - PyTorch backend will be used")


class LayoutPredictor:
    """
    Document layout prediction using safe tensors or ZDLC.
    
    Automatically uses ZDLC backend on s390x architecture, PyTorch on others.
    """

    def __init__(
        self,
        artifact_path: str,
        zdlc_model_path: Optional[str] = None,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Set[str] = set(),
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model configuration files.
        zdlc_model_path: Optional path to ZDLC compiled .so model file.
                        Required when running on s390x with ZDLC available.
        device: (Optional) device to run the inference (PyTorch backend only).
        num_threads: (Optional) Number of threads to run the inference.
        base_threshold: (Optional) Score threshold for predictions.
        blacklist_classes: (Optional) Set of class names to filter out.

        Raises
        ------
        FileNotFoundError when required model files are missing
        ValueError when zdlc_model_path is required but not provided
        """
        # Blacklisted classes
        self._black_classes = blacklist_classes

        # Canonical classes
        self._labels = LayoutLabels()

        # Set basic params
        self._threshold = base_threshold
        self._num_threads = num_threads
        self._model = None
        self._zdlc_session = None
        
        # Determine which backend to use
        use_zdlc = _IS_S390X and _ZDLC_AVAILABLE
        
        if use_zdlc:
            if zdlc_model_path is None:
                raise ValueError(
                    "zdlc_model_path is required when running on s390x with ZDLC available"
                )
            self._backend = "ZDLC"
            self._device = "cpu"  # ZDLC runs on CPU
            self._init_zdlc(artifact_path, zdlc_model_path)
        else:
            self._backend = "PyTorch"
            self._device = device
            self._init_pytorch(artifact_path, device)
        
        _log.info(f"LayoutPredictor initialized with {self._backend} backend")
        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def _init_pytorch(self, artifact_path: str, device: str):
        """Initialize PyTorch backend."""
        # Set device
        self._device_obj = torch.device(device)
        if device == "cpu":
            torch.set_num_threads(self._num_threads)

        # Load model file and configurations
        self._processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        self._model_config = os.path.join(artifact_path, "config.json")
        self._st_fn = os.path.join(artifact_path, "model.safetensors")
        
        if not os.path.isfile(self._st_fn):
            raise FileNotFoundError("Missing safe tensors file: {}".format(self._st_fn))
        if not os.path.isfile(self._processor_config):
            raise FileNotFoundError(
                f"Missing processor config file: {self._processor_config}"
            )
        if not os.path.isfile(self._model_config):
            raise FileNotFoundError(f"Missing model config file: {self._model_config}")

        # Load image processor
        self._image_processor = RTDetrImageProcessor.from_json_file(
            self._processor_config
        )

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            self._model = AutoModelForObjectDetection.from_pretrained(
                artifact_path, config=self._model_config, device_map=self._device_obj
            )
            self._model.eval()

        # Set classes map
        self._model_name = type(self._model).__name__
        if self._model_name == "RTDetrForObjectDetection":
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0

    def _init_zdlc(self, artifact_path: str, zdlc_model_path: str):
        """Initialize ZDLC backend."""
        # Load configurations
        self._processor_config = os.path.join(
            artifact_path, "preprocessor_config.json"
        )
        self._model_config = os.path.join(artifact_path, "config.json")

        if not os.path.isfile(self._processor_config):
            raise FileNotFoundError(
                f"Missing processor config file: {self._processor_config}"
            )
        if not os.path.isfile(self._model_config):
            raise FileNotFoundError(
                f"Missing model config file: {self._model_config}"
            )

        # Load image processor
        self._image_processor = RTDetrImageProcessor.from_json_file(
            self._processor_config
        )

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            self._zdlc_session = zdlc_pyrt.InferenceSession(zdlc_model_path)

        # Set classes map - assuming RTDetr model type
        self._model_name = "RTDetrForObjectDetection"
        self._classes_map = self._labels.shifted_canonical_categories()
        self._label_offset = 1

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "backend": self._backend,
            "model_name": self._model_name,
            "device": self._device,
            "num_threads": self._num_threads,
            "image_size": self._image_processor.size,
            "threshold": self._threshold,
        }
        if self._backend == "PyTorch":
            info["safe_tensors_file"] = self._st_fn
        return info

    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence", "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        if self._backend == "PyTorch":
            return self._predict_pytorch(orig_img)
        else:
            return self._predict_zdlc(orig_img)

    def _predict_pytorch(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """PyTorch backend prediction."""
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        with torch.inference_mode():
            target_sizes = torch.tensor([page_img.size[::-1]])
            inputs = self._image_processor(images=[page_img], return_tensors="pt").to(
                self._device_obj
            )
            outputs = self._model(**inputs)
            results: List[Dict[str, Tensor]] = (
                self._image_processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=self._threshold,
                )
            )

        w, h = page_img.size
        result = results[0]
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score = float(score.item())

            label_id = int(label_id.item()) + self._label_offset
            label_str = self._classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self._black_classes:
                continue

            bbox_float = [float(b.item()) for b in box]
            l = min(w, max(0, bbox_float[0]))
            t = min(h, max(0, bbox_float[1]))
            r = min(w, max(0, bbox_float[2]))
            b = min(h, max(0, bbox_float[3]))
            yield {
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "label": label_str,
                "confidence": score,
            }

    def _predict_zdlc(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """ZDLC backend prediction."""
        # Convert to PIL Image if needed
        if isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img)
        elif isinstance(orig_img, Image.Image):
            page_img = orig_img
        else:
            raise ValueError(
                "Unsupported input format. Supported formats are PIL.Image.Image or numpy.ndarray."
            )

        # Prepare inputs
        target_sizes = np.array([page_img.size[::-1]], dtype=np.int64)
        inputs = self._image_processor(images=[page_img], return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)

        # Run ZDLC inference - model expects [pixel_values, target_sizes]
        outputs = self._zdlc_session.run([pixel_values, target_sizes])
        
        # ZDLC model outputs are already post-processed:
        # Output 0: labels (batch_size, num_queries) - int64
        # Output 1: boxes (batch_size, num_queries, 4) - float32, absolute coords
        # Output 2: scores (batch_size, num_queries) - float32
        pred_labels = outputs[0]  # (batch_size, num_queries)
        pred_boxes = outputs[1]   # (batch_size, num_queries, 4)
        pred_scores = outputs[2]  # (batch_size, num_queries)
        
        # Filter by threshold
        results = []
        for i in range(pred_labels.shape[0]):
            mask = pred_scores[i] > self._threshold
            results.append({
                'labels': pred_labels[i][mask],
                'boxes': pred_boxes[i][mask],
                'scores': pred_scores[i][mask]
            })

        # Format results
        result = results[0]
        h, w = page_img.size[::-1]

        predictions = []
        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            bbox_float = box.tolist() if hasattr(box, 'tolist') else box
            l = min(w, max(0, bbox_float[0]))
            t = min(h, max(0, bbox_float[1]))
            r = min(w, max(0, bbox_float[2]))
            b = min(h, max(0, bbox_float[3]))

            label_int = int(label.item() if hasattr(label, 'item') else label)
            label_str = self._classes_map.get(label_int - self._label_offset, "unknown")

            if label_str not in self._black_classes:
                predictions.append(
                    {
                        "label": label_str,
                        "confidence": float(score.item() if hasattr(score, 'item') else score),
                        "l": l,
                        "t": t,
                        "r": r,
                        "b": b,
                    }
                )

        return predictions


    def predict_batch(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """
        Batch prediction for multiple images - more efficient than calling predict() multiple times.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to process in a single batch

        Returns
        -------
        List[List[dict]]
            List of prediction lists, one per input image. Each prediction dict contains:
            "label", "confidence", "l", "t", "r", "b"
        """
        if not images:
            return []

        # Convert all images to RGB PIL format
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Not supported input image format")

        # Get target sizes for all images
        target_sizes = torch.tensor([img.size[::-1] for img in pil_images])

        # Process all images in a single batch
        inputs = self._image_processor(images=pil_images, return_tensors="pt").to(
            self._device
        )
        outputs = self._model(**inputs)

        # Post-process all results at once
        results_list: List[Dict[str, Tensor]] = (
            self._image_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self._threshold,
            )
        )

        # Convert results to standard format for each image
        all_predictions = []

        for img, results in zip(pil_images, results_list):
            w, h = img.size
            predictions = []

            for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                score = float(score.item())
                label_id = int(label_id.item()) + self._label_offset
                label_str = self._classes_map[label_id]

                # Filter out blacklisted classes
                if label_str in self._black_classes:
                    continue

                bbox_float = [float(b.item()) for b in box]
                l = min(w, max(0, bbox_float[0]))
                t = min(h, max(0, bbox_float[1]))
                r = min(w, max(0, bbox_float[2]))
                b = min(h, max(0, bbox_float[3]))

                predictions.append(
                    {
                        "l": l,
                        "t": t,
                        "r": r,
                        "b": b,
                        "label": label_str,
                        "confidence": score,
                    }
                )

            all_predictions.append(predictions)

        return all_predictions

    def _predict_batch_zdlc(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """ZDLC backend batch prediction."""
        # Convert all to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError(
                    "Unsupported input format. Supported formats are PIL.Image.Image or numpy.ndarray."
                )

        # Prepare batch inputs
        target_sizes = np.array([img.size[::-1] for img in pil_images], dtype=np.int64)
        inputs = self._image_processor(images=pil_images, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)

        # Run ZDLC inference - model expects [pixel_values, target_sizes]
        outputs = self._zdlc_session.run([pixel_values, target_sizes])
        
        # ZDLC model outputs are already post-processed
        pred_labels = outputs[0]
        pred_boxes = outputs[1]
        pred_scores = outputs[2]
        
        # Filter by threshold
        results_list = []
        for i in range(pred_labels.shape[0]):
            mask = pred_scores[i] > self._threshold
            results_list.append({
                'labels': pred_labels[i][mask],
                'boxes': pred_boxes[i][mask],
                'scores': pred_scores[i][mask]
            })

        # Format results for each image
        all_predictions = []
        for idx, result in enumerate(results_list):
            h, w = pil_images[idx].size[::-1]
            predictions = []

            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                bbox_float = box.tolist() if hasattr(box, 'tolist') else box
                l = min(w, max(0, bbox_float[0]))
                t = min(h, max(0, bbox_float[1]))
                r = min(w, max(0, bbox_float[2]))
                b = min(h, max(0, bbox_float[3]))

                label_int = int(label.item() if hasattr(label, 'item') else label)
                label_str = self._classes_map.get(
                    label_int - self._label_offset, "unknown"
                )

                if label_str not in self._black_classes:
                    predictions.append(
                        {
                            "label": label_str,
                            "confidence": float(score.item() if hasattr(score, 'item') else score),
                            "l": l,
                            "t": t,
                            "r": r,
                            "b": b,
                        }
                    )

            all_predictions.append(predictions)

        return all_predictions
