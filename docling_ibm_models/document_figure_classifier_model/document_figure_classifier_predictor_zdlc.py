import logging
import threading
from typing import List, Tuple, Union

import numpy as np
import torchvision.transforms as transforms
import zdlc_pyrt
from PIL import Image
from transformers import AutoConfig

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class DocumentFigureClassifierPredictorZDLC:
    r"""
    Model for classifying document figures using ZDLC compiled model.

    Classifies figures as 1 out of 16 possible classes.

    The classes are:
        1. "bar_chart"
        2. "bar_code"
        3. "chemistry_markush_structure"
        4. "chemistry_molecular_structure"
        5. "flow_chart"
        6. "icon"
        7. "line_chart"
        8. "logo"
        9. "map"
        10. "other"
        11. "pie_chart"
        12. "qr_code"
        13. "remote_sensing"
        14. "screenshot"
        15. "signature"
        16. "stamp"

    Attributes
    ----------
    _num_threads : int
        Number of threads used for inference.
    _zdlc_session : zdlc_pyrt.InferenceSession
        ZDLC inference session for running the compiled model.
    _image_processor : torchvision.transforms.Compose
        Processor for normalizing and preparing input images.
    _classes: List[str]:
        The classes used by the model.

    Methods
    -------
    __init__(artifacts_path, zdlc_model_path, num_threads)
        Initializes the DocumentFigureClassifierPredictorZDLC.
    info() -> dict:
        Retrieves configuration details.
    predict(images) -> List[List[Tuple[str, float]]]
        The confidence scores for the classification of each image.
    """

    def __init__(
        self,
        artifacts_path: str,
        zdlc_model_path: str,
        num_threads: int = 4,
    ):
        r"""
        Initializes the DocumentFigureClassifierPredictorZDLC.

        Parameters
        ----------
        artifacts_path : str
            Path to the directory containing config files.
        zdlc_model_path : str
            Path to the ZDLC compiled .so model file.
        num_threads : int, optional
            Number of threads for inference, by default 4.
        """
        self._num_threads = num_threads

        with _model_init_lock:
            # Initialize ZDLC inference session
            self._zdlc_session = zdlc_pyrt.InferenceSession(zdlc_model_path)

            # The model expects:
            # - NCHW format (batch, channels, height, width)
            # - ImageNet normalization (same as original PyTorch model)
            self._image_processor = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),  # Converts to [0-1] and NCHW format
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.47853944, 0.4732864, 0.47434163],
                    ),
                ]
            )

            config = AutoConfig.from_pretrained(artifacts_path)

        self._classes = list(config.id2label.values())
        self._classes.sort()

        _log.debug(
            "DocumentFigureClassifierModelZDLC settings: {}".format(
                self.info()
            )
        )

    def info(self) -> dict:
        """
        Retrieves configuration details.

        Returns
        -------
        dict
            A dictionary containing configuration details such as
            the number of threads used and the classes used by the model.
        """
        info = {
            "backend": "ZDLC",
            "num_threads": self._num_threads,
            "classes": self._classes,
        }
        return info

    def predict(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[Tuple[str, float]]]:
        r"""
        Performs inference on a batch of figures.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            A list of input images for inference. Each image can either
            be a PIL.Image.Image object or a NumPy array representing
            an image.

        Returns
        -------
        List[List[Tuple[str, float]]]
            A list of predictions for each input image. Each prediction
            is a list of tuples representing the predicted class and
            confidence score:
            - str: The predicted class name for the image.
            - float: The confidence score associated with the predicted
              class, ranging from 0 to 1.

            The predictions for each image are sorted in descending
            order of confidence.
        """
        rgb_images = []
        for image in images:
            if isinstance(image, Image.Image):
                rgb_images.append(image.convert("RGB"))
            elif isinstance(image, np.ndarray):
                rgb_images.append(Image.fromarray(image).convert("RGB"))
            else:
                raise TypeError(
                    "Supported input formats are PIL.Image.Image or "
                    "numpy.ndarray."
                )

        # (batch_size, 3, 224, 224)
        processed_images = [
            self._image_processor(image) for image in rgb_images
        ]
        # Convert to numpy array for ZDLC
        # torchvision transforms output torch tensors, convert to numpy
        numpy_images = np.stack(
            [img.numpy() for img in processed_images]
        ).astype(np.float32)

        # Run ZDLC inference
        outputs = self._zdlc_session.run([numpy_images])

        # Process outputs
        # outputs[0] should contain logits (batch_size, num_classes)
        logits = outputs[0]
        
        # Check if output is already probabilities (sum to ~1) or logits
        # If already probabilities, use directly; otherwise apply softmax
        first_sample_sum = np.sum(np.abs(logits[0]))
        
        if 0.99 < first_sample_sum < 1.01:
            # Already probabilities
            probs_batch = logits.tolist()
        else:
            # Apply softmax to logits
            exp_logits = np.exp(
                logits - np.max(logits, axis=1, keepdims=True)
            )
            probs_batch = (
                exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            ).tolist()

        predictions_batch = []
        for probs_image in probs_batch:
            # Handle case where model outputs more/fewer classes than expected
            num_classes = min(len(probs_image), len(self._classes))
            preds = [
                (self._classes[i], probs_image[i])
                for i in range(num_classes)
            ]
            preds.sort(key=lambda t: t[1], reverse=True)
            predictions_batch.append(preds)

        return predictions_batch

# Made with Bob
