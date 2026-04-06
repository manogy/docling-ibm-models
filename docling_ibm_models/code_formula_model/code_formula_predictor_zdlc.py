import logging
import threading
from typing import List, Optional, Union

import numpy as np
import zdlc_pyrt
from PIL import Image
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from docling_ibm_models.code_formula_model.models.sam_opt_image_processor import (
    SamOptImageProcessor,
)

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class StopOnString(StoppingCriteria):
    def __init__(self, tokenizer, stop_string):
        self.stop_token_ids = tokenizer.encode(stop_string, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        for sequence in input_ids:
            sequence_list = sequence.tolist()
            for i in range(len(sequence_list) - len(self.stop_token_ids) + 1):
                if (
                    sequence_list[i : i + len(self.stop_token_ids)]
                    == self.stop_token_ids
                ):
                    return True
        return False


class CodeFormulaPredictorZDLC:
    """
    Code and Formula Predictor using ZDLC compiled model.

    This class enables the prediction of code or LaTeX representations
    from input images of code snippets or mathematical formulas using
    ZDLC (IBM Z Deep Learning Compiler) compiled models.

    Attributes
    ----------
    _num_threads : int
        Number of threads used for inference.
    _tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for processing textual inputs to the model.
    _zdlc_session : zdlc_pyrt.InferenceSession
        ZDLC inference session for running the compiled model.
    _image_processor : transformers.ImageProcessor
        Processor for normalizing and preparing input images.
    """

    def __init__(
        self,
        artifacts_path: str,
        zdlc_model_path: str,
        num_threads: int = 4,
    ):
        """
        Initializes the CodeFormulaPredictorZDLC with the specified model artifacts.

        Parameters
        ----------
        artifacts_path : str
            Path to the directory containing the pretrained model files (tokenizer, image processor).
        zdlc_model_path : str
            Path to the ZDLC compiled .so model file.
        num_threads : int, optional
            Number of threads for inference, by default 4.
        """
        self._num_threads = num_threads

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            self._tokenizer = AutoTokenizer.from_pretrained(
                artifacts_path, use_fast=True, padding_side="left"
            )
            
            # Initialize ZDLC inference session
            self._zdlc_session = zdlc_pyrt.InferenceSession(zdlc_model_path)
            
            self._image_processor = SamOptImageProcessor.from_pretrained(artifacts_path)

        _log.debug("CodeFormulaModelZDLC settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Retrieves configuration details of the CodeFormulaPredictorZDLC instance.

        Returns
        -------
        dict
            A dictionary containing configuration details such as the number of threads used.
        """
        info = {
            "backend": "ZDLC",
            "num_threads": self._num_threads,
        }
        return info

    def _get_prompt(self, label: str) -> str:
        """
        Constructs the prompt for the model based on the input label.

        Parameters
        ----------
        label : str
            The type of input, either 'code' or 'formula'.

        Returns
        -------
        str
            The constructed prompt including necessary tokens and query.

        Raises
        ------
        NotImplementedError
            If the label is not 'code' or 'formula'.
        """
        if label == "code":
            query = "<code_image_to_text>"
        elif label == "formula":
            query = "<equation>"
        else:
            raise NotImplementedError("Label must be either code or formula")

        prompt = (
            "A chat between a curious user and an artificial intelligence"
            " assistant. The assistant gives helpful, detailed, and polite answers to"
            " the user's questions. USER: "
        )
        prompt += (
            "<img>" + "<imgpad>" * 256 + "</img>" + "\n" + " ASSISTANT:" + "\n" + query
        )

        return prompt

    def _strip(self, text: str):
        """
        Removes any occurrences of the substrings in remove_list from the end of text.

        Parameters
        ----------
        text : str
            The original string.

        Returns
        -------
        str
            The trimmed string.
        """
        remove_list = [r"\quad", r"\\", r"\,", " c c c c", " l l l l l"]
        changed = True
        while changed:
            changed = False
            for substr in remove_list:
                if text.endswith(substr):
                    text = text[: -len(substr)]
                    changed = True

        return text.strip()

    def predict(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        labels: List[str],
        temperature: Optional[float] = 0.0,
    ) -> List[str]:
        """
        Predicts the textual representation of input images (code or LaTeX).

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to be processed, provided as PIL Image objects or numpy arrays.
        labels : List[str]
            List of labels indicating the type of each image ('code' or 'formula').
        temperature : Optional[float]
            Sampling temperature for generation, by default set to 0.0.

        Returns
        -------
        List[str]
            List of predicted textual outputs for each input image in the given input order.

        Raises
        ------
        TypeError
            If any of the input images is not of a supported type (PIL Image or numpy array).
        Exception
            In case the temperature is an invalid number.
        """
        if (
            temperature is None
            or not (isinstance(temperature, float) or isinstance(temperature, int))
            or temperature < 0
        ):
            raise Exception("Temperature must be a number greater or equal to 0.")

        if len(labels) != len(images):
            raise Exception(
                "The number of images must be the same as the number of labels."
            )

        images_tmp = []
        for image in images:
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            else:
                raise TypeError("Not supported input image format")
            images_tmp.append(image)

        # Process images to numpy arrays
        images_array = np.stack(
            [self._image_processor(img).numpy() for img in images_tmp]
        )

        prompts = [self._get_prompt(label) for label in labels]

        tokenized = self._tokenizer(prompts, padding=True, return_tensors="np")
        
        prompt_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Run ZDLC inference
        # Note: The exact input format depends on how the ONNX model was exported
        # This assumes the model takes input_ids, attention_mask, and images
        outputs = self._zdlc_session.run([prompt_ids, attention_mask, images_array])
        
        # Process outputs - assuming outputs[0] contains the generated token ids
        output_ids_list = outputs[0]
        
        # Decode the outputs
        outputs = self._tokenizer.batch_decode(
            output_ids_list[:, prompt_ids.shape[1] :], skip_special_tokens=True
        )
        outputs = [self._strip(output) for output in outputs]

        return outputs

# Made with Bob
