import logging
import os
import threading
from collections.abc import Iterable
from typing import Dict, List, Set, Union

import numpy as np
import zdlc_pyrt
from PIL import Image
from transformers import RTDetrImageProcessor

from docling_ibm_models.layoutmodel.labels import LayoutLabels

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class LayoutPredictorZDLC:
    """
    Document layout prediction using ZDLC compiled model
    """

    def __init__(
        self,
        artifact_path: str,
        zdlc_model_path: str,
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Set[str] = set(),
    ):
        """
        Provide the artifact path and ZDLC model path

        Parameters
        ----------
        artifact_path: Path for the model configuration files.
        zdlc_model_path: Path to the ZDLC compiled .so model file.
        num_threads: (Optional) Number of threads to run the inference
        base_threshold: (Optional) Score threshold for predictions
        blacklist_classes: (Optional) Set of class names to filter out

        Raises
        ------
        FileNotFoundError when required configuration files are missing
        """
        # Blacklisted classes
        self._black_classes = blacklist_classes

        # Canonical classes
        self._labels = LayoutLabels()

        # Set basic params
        self._threshold = base_threshold  # Score threshold

        # Set number of threads
        self._num_threads = num_threads

        # Load model file and configurations
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
            self._zdlc_session = zdlc_pyrt.InferenceSession(
                zdlc_model_path
            )

        # Set classes map - assuming RTDetr model type
        self._model_name = "RTDetrForObjectDetection"
        self._classes_map = self._labels.shifted_canonical_categories()
        self._label_offset = 1

        _log.debug("LayoutPredictorZDLC settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictorZDLC
        """
        info = {
            "backend": "ZDLC",
            "model_name": self._model_name,
            "num_threads": self._num_threads,
            "image_size": self._image_processor.size,
            "threshold": self._threshold,
        }
        return info

    def predict(
        self, orig_img: Union[Image.Image, np.ndarray]
    ) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted
        bbox coords are provided as: [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or
                    numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence",
        "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        target_sizes = np.array([page_img.size[::-1]], dtype=np.int64)
        inputs = self._image_processor(
            images=[page_img], return_tensors="np"
        )

        # Run ZDLC inference
        # Assuming the model expects pixel_values as input
        pixel_values = inputs["pixel_values"].astype(np.float32)
        outputs = self._zdlc_session.run([pixel_values])

        # Post-process outputs
        # Assuming outputs format: [logits, pred_boxes]
        # This may need adjustment based on actual ONNX model output
        logits = outputs[0]  # (batch_size, num_queries, num_classes)
        pred_boxes = outputs[1]  # (batch_size, num_queries, 4)

        # Apply threshold and convert to final format
        results = self._post_process_object_detection(
            logits, pred_boxes, target_sizes, self._threshold
        )

        w, h = page_img.size
        result = results[0]
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            label_id = int(label_id) + self._label_offset
            label_str = self._classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self._black_classes:
                continue

            l = min(w, max(0, box[0]))
            t = min(h, max(0, box[1]))
            r = min(w, max(0, box[2]))
            b = min(h, max(0, box[3]))
            yield {
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "label": label_str,
                "confidence": score,
            }

    def predict_batch(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """
        Batch prediction for multiple images - more efficient than
        calling predict() multiple times.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to process in a single batch

        Returns
        -------
        List[List[dict]]
            List of prediction lists, one per input image. Each
            prediction dict contains: "label", "confidence",
            "l", "t", "r", "b"
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
        target_sizes = np.array(
            [img.size[::-1] for img in pil_images], dtype=np.int64
        )

        # Process all images in a single batch
        inputs = self._image_processor(
            images=pil_images, return_tensors="np"
        )

        # Run ZDLC inference
        pixel_values = inputs["pixel_values"].astype(np.float32)
        outputs = self._zdlc_session.run([pixel_values])

        # Post-process outputs
        logits = outputs[0]
        pred_boxes = outputs[1]

        results_list = self._post_process_object_detection(
            logits, pred_boxes, target_sizes, self._threshold
        )

        # Convert results to standard format for each image
        all_predictions = []

        for img, results in zip(pil_images, results_list):
            w, h = img.size
            predictions = []

            for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                label_id = int(label_id) + self._label_offset
                label_str = self._classes_map[label_id]

                # Filter out blacklisted classes
                if label_str in self._black_classes:
                    continue

                l = min(w, max(0, box[0]))
                t = min(h, max(0, box[1]))
                r = min(w, max(0, box[2]))
                b = min(h, max(0, box[3]))

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

    def _post_process_object_detection(
        self, logits, pred_boxes, target_sizes, threshold
    ):
        """
        Post-process model outputs to get final predictions

        Parameters
        ----------
        logits : np.ndarray
            Model logits output (batch_size, num_queries, num_classes)
        pred_boxes : np.ndarray
            Predicted boxes (batch_size, num_queries, 4)
        target_sizes : np.ndarray
            Target image sizes (batch_size, 2)
        threshold : float
            Confidence threshold

        Returns
        -------
        List[Dict]
            List of predictions for each image
        """
        # Apply softmax to logits
        exp_logits = np.exp(
            logits - np.max(logits, axis=-1, keepdims=True)
        )
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Get scores and labels
        scores = np.max(probs, axis=-1)
        labels = np.argmax(probs, axis=-1)

        # Convert boxes from normalized to absolute coordinates
        results = []
        for i in range(len(logits)):
            # Filter by threshold
            keep = scores[i] > threshold
            filtered_scores = scores[i][keep]
            filtered_labels = labels[i][keep]
            filtered_boxes = pred_boxes[i][keep]

            # Scale boxes to image size
            img_h, img_w = target_sizes[i]
            # Assuming boxes are in cxcywh format, convert to xyxy
            cx, cy, w, h = (
                filtered_boxes[:, 0],
                filtered_boxes[:, 1],
                filtered_boxes[:, 2],
                filtered_boxes[:, 3],
            )
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h

            scaled_boxes = np.stack([x1, y1, x2, y2], axis=1)

            results.append(
                {
                    "scores": filtered_scores.tolist(),
                    "labels": filtered_labels.tolist(),
                    "boxes": scaled_boxes.tolist(),
                }
            )

        return results

# Made with Bob
