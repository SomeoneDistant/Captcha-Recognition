# Captcha Recognition

This repository provides a Python implementation for recognizing captchas by segmenting characters from images and matching them with learned character patterns.

---

## Overview

The `Captcha` class loads captcha images from './sampleCaptchas' as training dataset, processes them to extract binary character segments, generates a mapping between these segments and corresponding characters, and finally identifies characters in new captcha images.

---

## Features

- Load captcha images along with corresponding metadata files.
- Convert images to binary maps using a configurable threshold.
- Generate character-to-pattern mappings from training data.
- Identify captchas in new images by matching segments against learned patterns.
- Save identified captcha texts into output files.

---

## Requirements

- Python 3.x
- OpenCV
- NumPy
- matplotlib

---

## Usage

1. Place your training captcha images in the folder structure:
` ./sampleCaptchas/input/*.jpg `
Each image should have:
  - A corresponding input text file (same filename, `.txt` extension) with pixel and height/width metadata.
  - A corresponding output text file containing the captcha label.

2. Run the script:
```bash
python captcha_recognition.py
```
Outputs are saved in ./output.

3. Example
```python
from captcha_solver import Captcha
captcha = Captcha(train_path='./sampleCaptchas', threshold=40)
captcha('./sampleCaptchas/input/input00.jpg')
```

---

## Notes
1. Images with missing or mismatched metadata will be excluded from training and used for validation instead.

2. The current implementation assumes captchas have exactly 5 characters.

3. The current implementation assumes the font and spacing is the same each time.

4. The current implementation assumes there is no skew in the structure of the characters.

5. Pixel data must be consistent across all three color channels (R=G=B).

6. More information please refer to `AI Technical test.docx`

---

## Contact
For questions or suggestions, please open an issue or contact the maintainer:
`Yao Yuan (distantyy@gmail.com)`
