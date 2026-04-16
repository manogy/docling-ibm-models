import glob
import logging
import os
import threading
from typing import List

import numpy as np
import zdlc_pyrt

from docling_ibm_models.tableformer.common import otsl_to_html
from docling_ibm_models.tableformer.data_management import transforms as T
from docling_ibm_models.tableformer.data_management.functional import load_word_map
from docling_ibm_models.tableformer.data_management.matching_post_processor import (
    MatchingPostProcessor,
)
from docling_ibm_models.tableformer.data_management.tf_cell_matcher import CellMatcher
from docling_ibm_models.tableformer.utils import utils as u
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


def otsl_sqr_chk(rs_list, logdebug=False):
    """Check OTSL sequence validity"""
    rs_list_split = rs_list.split()

    # Check if the sequence is square
    isSquare = True

    init_tag_len = len(rs_list_split)

    totcelnum = 0

    for ind, ln in enumerate(rs_list_split):
        if ln == "<td":
            totcelnum += 1
        if ln == "</td>":
            totcelnum -= 1
            if totcelnum < 0:
                isSquare = False
                if logdebug:
                    logger.debug(
                        "OTSL sequence is not square at index {}".format(ind)
                    )
                break

    if totcelnum != 0:
        isSquare = False
        err_name = "OTSL sequence is not square"
        if logdebug:
            logger.debug(err_name)

    return isSquare


class TFPredictorZDLC:
    r"""
    Table predictions using ZDLC compiled model
    """

    def __init__(self, config, zdlc_model_path: str, num_threads: int = 4):
        r"""
        Parameters
        ----------
        config : dict Parameters configuration
        zdlc_model_path: Path to the ZDLC compiled .so model file
        num_threads: (Optional) Number of threads to run the inference

        Raises
        ------
        ValueError
        When the model cannot be found
        """
        self._num_threads = num_threads
        logger.info("Running with ZDLC backend")

        self._config = config
        self.enable_post_process = True

        self._padding = config["predict"].get("padding", False)
        self._padding_size = config["predict"].get("padding_size", 10)

        self._cell_matcher = CellMatcher(config)
        self._post_processor = MatchingPostProcessor(config)

        self._init_word_map()

        # Load the ZDLC model
        with _model_init_lock:
            self._zdlc_session = zdlc_pyrt.InferenceSession(
                zdlc_model_path
            )

        self._prof = config["predict"].get("profiling", False)
        self._profiling_agg_window = config["predict"].get(
            "profiling_agg_window", None
        )
        if self._profiling_agg_window is not None:
            AggProfiler(self._profiling_agg_window)
        else:
            AggProfiler()

        self._model_type = self._config["model"]["type"]
        self._remove_padding = False
        if self._model_type == "TableModel02":
            self._remove_padding = True

    def _init_word_map(self):
        r"""
        Initialize the word map for the model
        """
        self._prepared_data_dir = self._config["dataset"]["prepared_data_dir"]

        # Load word map
        self._word_map = None
        if self._prepared_data_dir is not None:
            data_name = self._config["dataset"]["data_name"]
            word_map_fn = os.path.join(
                self._prepared_data_dir, "WORDMAP_" + data_name + ".json"
            )

            with open(word_map_fn, "r") as f:
                self._word_map = load_word_map(f)

        self._init_data = {"word_map": self._word_map}

        self._rev_word_map = {v: k for k, v in self._word_map.items()}

    def get_init_data(self):
        return self._init_data

    def get_device(self):
        return "zdlc"

    def get_model_type(self):
        return self._model_type

    def _log(self):
        return logger

    def _deletebbox(self, listofbboxes, index):
        newlist = []
        for i, bbox in enumerate(listofbboxes):
            if i != index:
                newlist.append(bbox)
        return newlist

    def _remove_bbox_span_desync(self, prediction):
        # Remove bboxes that don't have corresponding spans
        index_to_delete_from = -1
        indexes_to_delete = []
        newbboxes = []
        for html_elem in prediction["rs_seq"].split():
            if html_elem == "<td":
                index_to_delete_from += 1
            elif html_elem == "</td>":
                indexes_to_delete.append(index_to_delete_from)

        for i, bbox in enumerate(prediction["bboxes"]):
            if i not in indexes_to_delete:
                newbboxes.append(bbox)

        return newbboxes

    def _check_bbox_sync(self, prediction):
        bboxes = prediction["bboxes"]
        match = True

        count_bbox = len(bboxes)

        count_td = 0
        for html_elem in prediction["rs_seq"].split():
            if html_elem == "<td":
                count_td += 1

        if count_bbox != count_td:
            match = False
            logger.warning(
                "Number of bboxes ({}) does not match number of <td> "
                "tags ({})".format(count_bbox, count_td)
            )
            bboxes = self._remove_bbox_span_desync(prediction)

        return match, bboxes

    def page_coords_to_table_coords(
        self, bbox, table_bbox, im_width, im_height
    ):
        """
        Convert page coordinates to table coordinates
        """
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        # Table bbox coordinates
        t_x1 = table_bbox[0]
        t_y1 = table_bbox[1]
        t_x2 = table_bbox[2]
        t_y2 = table_bbox[3]

        # Table dimensions
        tw = t_x2 - t_x1
        th = t_y2 - t_y1
        new_bbox = [
            (x1 - t_x1) / tw * im_width,
            (y1 - t_y1) / th * im_height,
            (x2 - t_x1) / tw * im_width,
            (y2 - t_y1) / th * im_height,
        ]
        return new_bbox

    def _depad_bboxes(self, bboxes, new_image_ratio):
        """
        Remove padding from bounding boxes
        """
        new_bboxes = []
        c_x = 0.5
        c_y = 0.5

        for bbox in bboxes:
            # Convert from normalized coordinates
            cb_x1 = bbox[0]
            cb_y1 = bbox[1]
            cb_x2 = bbox[2]
            cb_y2 = bbox[3]

            # Remove padding effect
            r_cb_x1 = (cb_x1 - c_x) / new_image_ratio + c_x
            r_cb_y1 = (cb_y1 - c_y) / new_image_ratio + c_y
            r_cb_x2 = (cb_x2 - c_x) / new_image_ratio + c_x
            r_cb_y2 = (cb_y2 - c_y) / new_image_ratio + c_y

            # Clip to valid range
            x1 = max(0.0, min(1.0, r_cb_x1))
            y1 = max(0.0, min(1.0, r_cb_y1))
            x2 = max(0.0, min(1.0, r_cb_x2))
            y2 = max(0.0, min(1.0, r_cb_y2))

            new_bbox = [x1, y1, x2, y2]
            new_bboxes.append(new_bbox)

        return new_bboxes

    def _merge_tf_output(self, docling_output, pdf_cells):
        """
        Merge tableformer output with PDF cells
        """
        tf_output = []
        tf_cells_map = {}
        max_row_idx = 0

        for docling_item in docling_output:
            r_idx = docling_item["row_ids"]
            c_idx = docling_item["column_ids"]
            cell_key = (r_idx[0], c_idx[0])

            pdf_cell = None
            text_cell_bbox = docling_item["bbox"]

            tf_cells_map[cell_key] = {
                "bbox": text_cell_bbox,
                "row_ids": r_idx,
                "column_ids": c_idx,
            }

            if r_idx[-1] > max_row_idx:
                max_row_idx = r_idx[-1]

        # Merge logic here (simplified)
        for k in tf_cells_map:
            tf_output.append(tf_cells_map[k])

        return tf_output

    def resize_img(self, image, width, height, inter=None):
        """
        Resize image to specified dimensions
        """
        dim = None
        h, w = image.shape[:2]
        sf = 1.0

        if width is None and height is None:
            return image, sf
        if width is None:
            sf = height / float(h)
            dim = (int(w * sf), height)
        else:
            sf = width / float(w)
            dim = (width, int(h * sf))

        resized = np.resize(image, dim)
        return resized, sf

    def predict(
        self,
        iocr_page,
        table_bbox,
        table_image,
        scale_factor,
        eval_res_preds=None,
        correct_overlapping_cells=False,
    ):
        r"""
        Predict the table out of an image in memory

        Parameters
        ----------
        iocr_page : dict
            Docling provided table data
        table_bbox : list
            Table bounding box coordinates
        table_image : np.ndarray
            Table image as numpy array
        scale_factor : float
            Scale factor for the image
        eval_res_preds : dict
            Ready predictions provided by the evaluation results
        correct_overlapping_cells : boolean
            Enables or disables last post-processing step

        Returns
        -------
        tf_output : list
            Table structure output
        matching_details : dict
            Details about the matching between PDF and table cells
        """
        AggProfiler().start_agg(self._prof)

        max_steps = self._config["predict"]["max_steps"]
        beam_size = self._config["predict"]["beam_size"]
        image_batch = self._prepare_image(table_image)

        prediction = {}

        # Run ZDLC inference
        if eval_res_preds is not None:
            prediction["bboxes"] = eval_res_preds["bboxes"]
            pred_tag_seq = eval_res_preds["tag_seq"]
        elif self._config["predict"]["bbox"]:
            # Run model inference
            outputs = self._zdlc_session.run([image_batch])

            # Process outputs
            # Assuming outputs: [tag_seq, class_logits, coord_predictions]
            pred_tag_seq = outputs[0]
            outputs_class = outputs[1] if len(outputs) > 1 else None
            outputs_coord = outputs[2] if len(outputs) > 2 else None

            if outputs_coord is not None:
                if len(outputs_coord) == 0:
                    prediction["bboxes"] = []
                else:
                    bbox_pred = u.box_cxcywh_to_xyxy(outputs_coord)
                    prediction["bboxes"] = bbox_pred.tolist()
            else:
                prediction["bboxes"] = []

            if outputs_class is not None:
                if len(outputs_class) == 0:
                    prediction["classes"] = []
                else:
                    result_class = np.argmax(outputs_class, axis=1)
                    prediction["classes"] = result_class.tolist()
            else:
                prediction["classes"] = []

            if self._remove_padding:
                pred_tag_seq, _ = u.remove_padding(pred_tag_seq)
        else:
            outputs = self._zdlc_session.run([image_batch])
            pred_tag_seq = outputs[0]

            if self._remove_padding:
                pred_tag_seq, _ = u.remove_padding(pred_tag_seq)

        prediction["tag_seq"] = pred_tag_seq
        prediction["rs_seq"] = self._get_html_tags(pred_tag_seq)
        prediction["html_seq"] = otsl_to_html(prediction["rs_seq"], False)

        self._log().debug("----- rs_seq -----")
        self._log().debug(prediction["rs_seq"])
        self._log().debug(len(prediction["rs_seq"]))
        otsl_sqr_chk(prediction["rs_seq"], False)

        sync, corrected_bboxes = self._check_bbox_sync(prediction)
        if not sync:
            prediction["bboxes"] = corrected_bboxes

        # Match the cells
        matching_details = {
            "table_cells": [],
            "matches": {},
            "pdf_cells": [],
            "prediction_bboxes_page": [],
        }

        scaled_table_bbox = [
            table_bbox[0] / scale_factor,
            table_bbox[1] / scale_factor,
            table_bbox[2] / scale_factor,
            table_bbox[3] / scale_factor,
        ]

        if len(prediction["bboxes"]) > 0:
            matching_details = self._cell_matcher.match_cells(
                iocr_page, scaled_table_bbox, prediction
            )

        # Post-processing
        if len(prediction["bboxes"]) > 0:
            if len(iocr_page["tokens"]) > 0:
                if self.enable_post_process:
                    AggProfiler().begin("post_process", self._prof)
                    matching_details = self._post_processor.process(
                        matching_details, correct_overlapping_cells
                    )
                    AggProfiler().end("post_process", self._prof)

        # Generate the expected Docling responses
        AggProfiler().begin("generate_docling_response", self._prof)
        docling_output = self._generate_tf_response(
            matching_details["table_cells"],
            matching_details["matches"],
        )

        AggProfiler().end("generate_docling_response", self._prof)

        docling_output.sort(key=lambda item: item["cell_id"])
        matching_details["docling_responses"] = docling_output

        tf_output = self._merge_tf_output(
            docling_output, matching_details["pdf_cells"]
        )

        return tf_output, matching_details

    def _generate_tf_response(self, table_cells, matches):
        """
        Generate tableformer response format
        """
        tf_cell_list = []

        for pdf_cell_id, pdf_cell_matches in matches.items():
            tf_cell = {}

            row_ids = []
            column_ids = []
            labels = []

            for match in pdf_cell_matches:
                tm = match["table_match"]
                tcl = match["table_cell"]

                table_cell = table_cells[tcl]

                row_ids.extend(table_cell["row_ids"])
                column_ids.extend(table_cell["column_ids"])
                labels.extend(table_cell["labels"])

            tf_cell["cell_id"] = pdf_cell_id
            tf_cell["row_ids"] = sorted(list(set(row_ids)))
            tf_cell["column_ids"] = sorted(list(set(column_ids)))
            tf_cell["bbox"] = table_cells[0]["bbox"]

            tf_cell_list.append(tf_cell)

        return tf_cell_list

    def _prepare_image(self, mat_image):
        r"""
        Rescale the image and prepare a batch with the image as tensor

        Parameters
        ----------
        mat_image: np.ndarray
            The image as a numpy array

        Returns
        -------
        np.ndarray (batch_size, image_channels, resized_image,
                    resized_image)
        """
        normalize = T.Normalize(
            mean=self._config["dataset"]["image_normalization"]["mean"],
            std=self._config["dataset"]["image_normalization"]["std"],
        )
        resized_size = self._config["dataset"]["resized_image"]
        resize = T.Resize([resized_size, resized_size])

        img, _ = normalize(mat_image, None)
        img, _ = resize(img, None)

        img = img.transpose(2, 1, 0)  # (channels, width, height)
        img = img / 255.0
        image_batch = np.expand_dims(img, axis=0).astype(np.float32)

        return image_batch

    def _get_html_tags(self, seq):
        r"""
        Convert indices to actual html tags
        """
        # Map the tag indices back to actual tags (without start, end)
        html_tags = [self._rev_word_map[ind] for ind in seq[1:-1]]

        return html_tags

# Made with Bob
