"""Prepare official MMTab-eval without passing reference answers to agents."""
from collections import Counter
import hashlib
import json
from pathlib import Path

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}


def image_key(value):
    name = Path(str(value)).name
    # TabFact IDs legitimately end with .html.csv; only strip IMAGE extensions.
    return Path(name).stem if Path(name).suffix.lower() in IMAGE_SUFFIXES else name


def prepare_manifest(dataset, images, output, include=(), exclude=(), limit=0, shard_index=0, shards=1):
    dataset, images, output = Path(dataset).resolve(), Path(images).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Manifest already exists; choose a new output path to preserve the frozen selection')
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError('Require 0 <= shard-index < shards')
    image_index = {}
    for path in sorted(images.rglob('*')):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            if path.stem in image_index:
                raise ValueError('Ambiguous image ID; point --images to one image variant: ' + path.stem)
            image_index[path.stem] = path
    samples = json.loads(dataset.read_text(encoding='utf-8-sig'))
    selected = []
    for item in samples:
        task = item['task_type']
        if (include and task not in include) or task in exclude:
            continue
        selected.append(item)
    if limit:
        selected = selected[:limit]
    selected = selected[shard_index::shards]
    if not selected:
        raise ValueError('Empty evaluation selection')
    absent = [str(item['image_id']) for item in selected if image_key(item['image_id']) not in image_index]
    if absent:
        raise ValueError(f'{len(absent)} selected samples have no image: {absent[:8]}; no partial manifest written')
    rows, missing, hashes = [], [], {}
    for item in selected:
        image_id = str(item['image_id'])
        image = image_index.get(image_key(image_id))
        if image is None:
            missing.append(image_id)
            continue
        if image_id not in hashes:
            hashes[image_id] = hashlib.sha256(image.read_bytes()).hexdigest()
        rows.append({'id': item['item_id'], 'image_id': image_id, 'question': item['input'],
                     'image': str(image), 'image_sha256': hashes[image_id],
                     'task_type': item['task_type'], 'dataset_name': item['dataset_name']})
    if missing:
        raise ValueError(f'{len(missing)} selected samples have no image: {missing[:8]}; no partial manifest written')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation item IDs')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {'source_dataset': str(dataset), 'source_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                'image_root': str(images), 'count': len(rows), 'task_counts': dict(Counter(r['task_type'] for r in rows)),
                'filters': {'include': list(include), 'exclude': list(exclude), 'limit_before_sharding': limit,
                            'shard_index': shard_index, 'shards': shards},
                'note': 'No reference answers in inference manifest. Image variants must not be mixed.'}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata
