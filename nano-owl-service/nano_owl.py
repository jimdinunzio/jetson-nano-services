#!/usr/bin/env python3
"""
NanoOwl - Open-vocabulary object detection using NVIDIA NanoOWL tree predictor.

Provides a threaded, controllable interface for running open-vocabulary
object detection on video frames.

Frame source is chosen at init: "camera" for local USB camera, or "network"
for frames pushed over the wire via push_frame().

Supports enable/disable for stopping captures and reducing resource usage.
Auto-disables after 30s of no get_detections() calls.
"""

import io
import time
import threading
import cv2
import PIL.Image
from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.owl_predictor import OwlPredictor


class NanoOwl:
    """
    Threaded open-vocabulary object detection using NanoOWL.

    Run in a separate thread with start_up() as the entry point.
    Starts in disabled mode - call enable() to begin processing.

    Frame source is set at init and fixed for the lifetime of the instance:
      - "camera": captures from a local video device
      - "network": frames are pushed via push_frame()
    """

    AUTO_DISABLE_TIMEOUT = 30.0

    def __init__(self, image_encode_engine, frame_source="camera",
                 video_input=0, resolution=(640, 480)):
        """
        Args:
            image_encode_engine: Path to the TensorRT engine for OWL image encoder.
            frame_source: "camera" for local USB camera, "network" for pushed frames.
            video_input: Camera device index or video source path (camera mode only).
            resolution: Tuple of (width, height) for camera resolution (camera mode only).
        """
        if frame_source not in ("camera", "network"):
            raise ValueError(f"frame_source must be 'camera' or 'network', got '{frame_source}'")

        self.image_encode_engine = image_encode_engine
        self._frame_source = frame_source
        self.video_input = video_input
        self.resolution = resolution

        # Predictor (initialized in start_up)
        self._predictor = None

        # Prompt state
        self._prompt_data = None
        self._prompt_string = ""

        # Latest detections result
        self._detections = None
        self._detections_ready = threading.Event()

        # Threading control
        self._run_flag = False
        self._lock = threading.Lock()
        self._enabled = False
        self._enabled_event = threading.Event()

        # Auto-disable tracking
        self._last_get_time = time.time()

        # Camera (camera mode only, initialized in start_up)
        self._camera = None

        # Network frame buffer (network mode only)
        self._network_frame = None
        self._network_frame_seq = None
        self._network_frame_size = None  # (w, h), set from first pushed frame
        self._network_frame_event = threading.Event()

    def get_frame_source(self):
        return self._frame_source

    def enable(self):
        """Enable detection processing."""
        with self._lock:
            self._enabled = True
            self._last_get_time = time.time()
        self._enabled_event.set()

    def disable(self):
        """Disable detection processing and release resources."""
        with self._lock:
            self._enabled = False
            self._detections = None
        self._enabled_event.clear()
        self._detections_ready.clear()
        if self._frame_source == "camera":
            self._release_camera()
        else:
            with self._lock:
                self._network_frame = None
                self._network_frame_seq = None
            self._network_frame_event.clear()

    def is_enabled(self):
        with self._lock:
            return self._enabled

    def push_frame(self, jpeg_bytes, seq):
        """
        Push a JPEG-encoded frame for detection (network mode only).

        Args:
            jpeg_bytes: Raw JPEG bytes of the frame.
            seq: Integer sequence number assigned by the client.

        Returns:
            True on success, False if not in network mode or decode fails.
        """
        if self._frame_source != "network":
            return False
        try:
            frame = PIL.Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        except Exception as e:
            print(f"[NanoOwl] Error decoding pushed frame: {e}", flush=True)
            return False

        with self._lock:
            if self._network_frame_size is None:
                self._network_frame_size = frame.size
                print(f"[NanoOwl] Network frame size set to {frame.size}", flush=True)
            self._network_frame = frame
            self._network_frame_seq = seq
            self._last_get_time = time.time()
            if not self._enabled:
                self._enabled = True
        self._network_frame_event.set()
        self._enabled_event.set()
        return True

    def set_prompt(self, prompt):
        """
        Set the NanoOWL tree prompt and pre-encode text embeddings.

        The prompt uses NanoOWL tree syntax, e.g.:
            "[a person [a face, a hand], a dog]"

        Auto-enables processing if currently disabled.

        Returns:
            True on success, False on error or if predictor not loaded.
        """
        if self._predictor is None:
            return False
        try:
            tree = Tree.from_prompt(prompt)
            clip_encodings = self._predictor.encode_clip_text(tree)
            owl_encodings = self._predictor.encode_owl_text(tree)
        except Exception as e:
            print(f"[NanoOwl] Error setting prompt: {e}", flush=True)
            return False

        with self._lock:
            self._prompt_data = {
                "tree": tree,
                "clip_encodings": clip_encodings,
                "owl_encodings": owl_encodings,
            }
            self._prompt_string = prompt
            self._detections = None
            self._last_get_time = time.time()
            if not self._enabled:
                self._enabled = True
        self._detections_ready.clear()
        self._enabled_event.set()
        return True

    def get_prompt(self):
        with self._lock:
            return self._prompt_string

    def get_detections(self):
        """
        Get the latest detections (non-blocking).

        Resets the auto-disable timer.

        Returns:
            Dict with keys:
                - prompt: the current prompt string
                - detections: list of detection dicts
                - inference_time: seconds for the prediction call
                - frame_seq: sequence number of the frame (network mode, None for camera)
            Returns None if no detections are available yet.
        """
        with self._lock:
            self._last_get_time = time.time()
            if self._detections is None:
                return None
            return dict(self._detections)

    def _check_auto_disable(self):
        with self._lock:
            if not self._enabled:
                return False
            if time.time() - self._last_get_time >= self.AUTO_DISABLE_TIMEOUT:
                self._enabled = False
                self._detections = None
                print("[NanoOwl] Auto-disabled due to inactivity", flush=True)
        if not self._enabled:
            self._enabled_event.clear()
            self._detections_ready.clear()
            if self._frame_source == "camera":
                self._release_camera()
            else:
                with self._lock:
                    self._network_frame = None
                    self._network_frame_seq = None
                self._network_frame_event.clear()
            return True
        return False

    def _open_camera(self):
        """Open the video capture device."""
        camera = cv2.VideoCapture(self.video_input)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        return camera

    def _capture_frame(self):
        """
        Capture a frame from the configured source.

        Returns:
            (PIL.Image, seq) tuple. seq is an int for network mode, None for camera.
            Returns (None, None) if no frame is available.
        """
        if self._frame_source == "network":
            with self._lock:
                frame = self._network_frame
                seq = self._network_frame_seq
                self._network_frame = None
                self._network_frame_seq = None
            self._network_frame_event.clear()
            return frame, seq
        else:
            if self._camera is None:
                return None, None
            ret, frame = self._camera.read()
            if not ret:
                return None, None
            return PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), None

    def _release_camera(self):
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    @staticmethod
    def _serialize_tree_output(tree_output, tree, image_size):
        """
        Convert NanoOWL tree output to a serializable list of detection dicts.

        Each TreeDetection has: box, id, labels, parent_id, scores.
        """
        w, h = image_size
        results = []
        for det in tree_output.detections:
            box = det.box
            if hasattr(box, 'tolist'):
                box = box.tolist()
            elif not isinstance(box, list):
                box = list(box)
            # Clamp to image bounds
            box = [
                max(0.0, min(box[0], float(w))),
                max(0.0, min(box[1], float(h))),
                max(0.0, min(box[2], float(w))),
                max(0.0, min(box[3], float(h))),
            ]

            scores = det.scores
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = list(scores)

            labels = det.labels
            if hasattr(labels, 'tolist'):
                labels = labels.tolist()
            elif not isinstance(labels, list):
                labels = list(labels)

            label_id = int(det.id) if det.id is not None else -1
            parent_id = int(det.parent_id) if det.parent_id is not None else -1

            results.append({
                "label": tree.labels[label_id] if 0 <= label_id < len(tree.labels) else "",
                "label_id": label_id,
                "parent_label": tree.labels[parent_id] if 0 <= parent_id < len(tree.labels) else "",
                "parent_id": parent_id,
                "box": box,
                "scores": scores,
                "labels": labels,
            })
        return results

    def _get_image_size(self, frame):
        """Get (width, height) from frame, appropriate to the frame source."""
        if self._frame_source == "network":
            with self._lock:
                size = self._network_frame_size
            return size if size is not None else frame.size
        else:
            return self.resolution

    def start_up(self):
        """
        Main processing loop - designed to be run in a separate thread.

        Loads the predictor, then continuously captures frames and runs
        detection when enabled.
        """
        self._run_flag = True

        print("[NanoOwl] Loading predictor...", flush=True)
        self._predictor = TreePredictor(
            owl_predictor=OwlPredictor(
                image_encoder_engine=self.image_encode_engine
            )
        )
        print("[NanoOwl] Predictor loaded.", flush=True)

        print(f"[NanoOwl] Frame source: {self._frame_source}", flush=True)
        print("[NanoOwl] Starting in disabled mode.", flush=True)

        while self._run_flag:
            self._check_auto_disable()

            with self._lock:
                enabled = self._enabled
            if not enabled:
                self._enabled_event.wait(timeout=0.5)
                continue

            # Open camera if needed (camera mode only)
            if self._frame_source == "camera" and self._camera is None:
                self._camera = self._open_camera()
                print("[NanoOwl] Camera opened.", flush=True)

            frame, seq = self._capture_frame()
            if frame is None:
                if self._frame_source == "network":
                    self._network_frame_event.wait(timeout=0.5)
                continue

            with self._lock:
                prompt_data = self._prompt_data
            if prompt_data is None:
                time.sleep(0.1)
                continue

            try:
                t0 = time.perf_counter_ns()
                tree_output = self._predictor.predict(
                    frame,
                    tree=prompt_data['tree'],
                    clip_text_encodings=prompt_data['clip_encodings'],
                    owl_text_encodings=prompt_data['owl_encodings'],
                )
                t1 = time.perf_counter_ns()
                dt = (t1 - t0) / 1e9

                image_size = self._get_image_size(frame)
                det_list = NanoOwl._serialize_tree_output(
                    tree_output, prompt_data['tree'], image_size
                )

                with self._lock:
                    self._detections = {
                        "prompt": self._prompt_string,
                        "detections": det_list,
                        "inference_time": dt,
                        "frame_seq": seq,
                    }
                self._detections_ready.set()
            except Exception as e:
                print(f"[NanoOwl] Error in detection loop: {e}", flush=True)
                import traceback
                traceback.print_exc()

        self._release_camera()
        self._run_flag = False

    def shut_down(self):
        """Signal the processing loop to stop."""
        self._run_flag = False
        self._enabled_event.set()
        self._network_frame_event.set()

    def is_running(self):
        return self._run_flag
