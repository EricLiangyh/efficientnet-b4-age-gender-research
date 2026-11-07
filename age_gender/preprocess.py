"""Corrected face alignment for new experiments; see docs/provenance.md."""
import argparse
import csv
import hashlib
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def align_crop(image, face, padding=0.35, size=380):
    left, right = sorted(np.asarray(face.kps[:2]), key=lambda p: p[0])
    angle = np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))
    center = tuple(((left + right) / 2).astype(float))
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    h, w = image.shape[:2]
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC)
    x1, y1, x2, y2 = face.bbox
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]) @ matrix.T
    lo, hi = corners.min(0), corners.max(0)
    extent = hi - lo
    lo = np.maximum(0, np.floor(lo - padding * extent)).astype(int)
    hi = np.minimum([w, h], np.ceil(hi + padding * extent)).astype(int)
    crop = rotated[lo[1]:hi[1], lo[0]:hi[0]]
    if not crop.size:
        raise ValueError('Empty aligned face crop')
    return cv2.resize(crop, (size, size))


def main():
    p = argparse.ArgumentParser(description='Align labelled images from a CSV: path,age,gender,source')
    p.add_argument('--index', required=True)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--provider', default='CPUExecutionProvider')
    p.add_argument('--padding', type=float, default=0.35)
    p.add_argument('--size', type=int, default=380)
    args = p.parse_args()
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(allowed_modules=['detection'], providers=[args.provider])
    app.prepare(ctx_id=0 if 'CUDA' in args.provider else -1, det_size=(640, 640))
    root, output = Path(args.raw_root).resolve(), Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with open(args.index, encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    report = []
    for row in tqdm(rows):
        status = {'path': row['path'], 'status': 'error', 'output': ''}
        try:
            path = (root / row['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Source outside data root')
            if row['source'] not in {'UTKFace', 'AFAD', 'imdb'}:
                raise ValueError('Unknown source')
            age, gender = float(row['age']), int(row['gender'])
            if not 0 <= age <= 120 or gender not in {0, 1}:
                raise ValueError('Invalid labels')
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Unreadable image')
            faces = app.get(image)
            if len(faces) != 1:
                raise ValueError(f'Expected exactly one face; found {len(faces)}')
            crop = align_crop(image, faces[0], args.padding, args.size)
            suffix = hashlib.sha256(row['path'].encode()).hexdigest()[:20]
            dest = output / row['source'] / f'{age:g}_{gender}_{suffix}.jpg'
            dest.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode('.jpg', crop)
            if not ok:
                raise ValueError('Image encoding failed')
            encoded.tofile(dest)
            status.update(status='ok', output=dest.relative_to(output).as_posix())
        except Exception as error:
            status['status'] = str(error)
        report.append(status)
    with (output / 'preprocessing_report.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'status', 'output'])
        writer.writeheader()
        writer.writerows(report)
    print(f'Processed {sum(r["status"] == "ok" for r in report)}/{len(report)} images')


if __name__ == '__main__':
    main()
