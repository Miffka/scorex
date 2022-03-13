import re
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import pytesseract


@dataclass
class PlayerScore:
    name: str
    score: List[int]
    is_serving: bool


@dataclass
class TennisScore:
    player1: Optional[PlayerScore] = None
    player2: Optional[PlayerScore] = None

    def draw(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()

        for player, y_position in zip([self.player1, self.player2], [80, 140]):
            cv2.putText(
                img, str(int(player.is_serving)), (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4
            )
            cv2.putText(img, player.name, (100, y_position), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
            cv2.putText(
                img, "-".join(map(str, player.score)), (650, y_position), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4
            )

        return img


class ScoreParser:
    name_pat = r"([]>\*\.A-Za-z\ ]+)"
    clean_name_pat = r"[A-Z\.a-z]+"
    digit_pat = r"[0-9]+"
    serving_pat = r"[>\*]"

    def process_tesseract_data(self, data: pd.DataFrame):
        score_obj = TennisScore()

        for line_num in data.line_num.unique():
            line = data[data.line_num == line_num]

            name = ""
            digits = []
            for text in line["text"]:
                if re.match(self.name_pat, text):
                    name += f" {text}"

                if re.match(self.digit_pat, text):
                    digits.extend(map(int, re.findall(self.digit_pat, text)))

            is_serving = bool(re.search(self.serving_pat, name))
            name = " ".join(re.findall(self.clean_name_pat, name))
            player = PlayerScore(name, score=digits, is_serving=is_serving)

            if score_obj.player1 is None:
                score_obj.player1 = player
            else:
                score_obj.player2 = player

        return score_obj

    def find_text(self, img: np.ndarray) -> TennisScore:
        custom_config = r"--oem 1 --psm 3"

        # Convert image to grayscale and invert it
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = 255 - gray

        # Get all text from tesseract
        data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DATAFRAME)
        data = data[data.conf > 20].sort_values(["line_num", "left"])
        if data.shape[0] < 4:
            return None

        score_obj = self.process_tesseract_data(data)

        return score_obj
