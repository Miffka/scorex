import argparse
import os.path as osp

import cv2
import numpy as np

from scorex.config import system_config
from scorex.model.box import Box, get_score_bbox
from scorex.model.text import ScoreParser, TennisScore


class Predictor:
    def find_tableu(self, img: np.ndarray) -> Box:
        return get_score_bbox(img)

    def parse_tableu(self, img: np.ndarray) -> TennisScore:
        parser = ScoreParser()
        return parser.find_text(img)


class VideoPredictor:
    def __init__(self, video_path: str, update_rate: int = 20, visualize: bool = True):
        self.video_path = video_path
        self.update_rate = update_rate
        self.visualize = visualize

        self.predictor = Predictor()

    @staticmethod
    def img_difference(first: np.ndarray, second: np.ndarray) -> float:
        return (first - second).sum() / first.sum()

    def demo_run(self):
        vidcap = cv2.VideoCapture(self.video_path)
        success = True

        frame_idx = 0
        box = Box(0, 0, 0, 0)
        prev_cut_box = np.asarray(0)
        score_text = None

        while success:
            success, image = vidcap.read()
            image = image[:, :, ::-1]

            # Get cut box
            cut_box = box.apply(image)
            if frame_idx % self.update_rate == 0 or (
                not box.is_valid or self.img_difference(prev_cut_box, cut_box) > 0.3
            ):
                box = self.predictor.find_tableu(image)
                cut_box = box.apply(image)

                # Read text from box
                if box.is_valid:
                    score_text = self.predictor.parse_tableu(cut_box)
                else:
                    score_text = None

            if self.visualize:
                if score_text is not None:
                    img = box.draw(image)
                    img = score_text.draw(img)
                cv2.imshow("Frame", img[:, :])
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

            prev_cut_box = cut_box
            frame_idx += 1

        cv2.destroyAllWindows()
        vidcap.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Demo for tennis score parser")
    parser.add_argument(
        "--video_path",
        default=osp.join(system_config.data_dir, "raw", "top-100-shots-rallies-2018-atp-season.mp4"),
        help="Path to a video",
    )
    args = parser.parse_args()

    video_predictor = VideoPredictor(args.video_path)
    video_predictor.demo_run()
