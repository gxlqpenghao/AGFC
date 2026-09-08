from __future__ import annotations

import json
import socket
import threading
from http.server import HTTPServer
from io import BytesIO
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageChops

from agfc.adapters.mineru import repair_mineru_artifact
from agfc.core.export import export_figure_crops
from agfc.core.models import FigureCandidate
from agfc.core.object_export import extract_clean_figure_image
from agfc.runtime.cli import main
from agfc.runtime.output import prepare_output_dir
from agfc.runtime.runner import _render_page, run_pdf
from agfc.runtime.service import MAX_REQUEST_BYTES, _AgfcRequestHandler, dispatch_json_request


@pytest.mark.parametrize('pages', [[], [-1], [1], [True], [0, False]])
def test_invalid_pages_leave_output_untouched(tmp_path, pages):
    source = tmp_path / 'source.pdf'
    with fitz.open() as doc:
        doc.new_page()
        doc.save(source)
    output = tmp_path / 'out'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('keep')
    with pytest.raises(ValueError, match='page'):
        run_pdf(source, output_dir=output, pages=pages)
    assert sentinel.read_text() == 'keep'


def test_output_policy_preserves_input_and_unrelated_files(tmp_path):
    source = tmp_path / 'source.pdf'
    with fitz.open() as doc:
        doc.new_page()
        doc.save(source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match='new or empty'):
        run_pdf(source, output_dir=tmp_path)
    assert source.read_bytes() == before
    link = tmp_path / 'link'
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        prepare_output_dir(link)


def test_corrupt_pdf_does_not_create_output(tmp_path):
    source = tmp_path / 'bad.pdf'
    source.write_text('not a pdf')
    with pytest.raises(Exception):
        run_pdf(source, output_dir=tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_service_survives_missing_file_and_engine_exception(tmp_path, monkeypatch):
    assert dispatch_json_request('POST', '/extract', {'input': str(tmp_path / 'missing.pdf')})[0] == 400
    def fail(*args, **kwargs):
        raise RuntimeError('private backend detail')
    monkeypatch.setattr('agfc.runtime.service.run_pdf', fail)
    status, payload = dispatch_json_request('POST', '/extract', {'input': 'exists.pdf'})
    assert status == 500
    assert 'private backend detail' not in json.dumps(payload)
    assert dispatch_json_request('GET', '/health', None)[0] == 200


@pytest.mark.parametrize('headers,body,expected', [
    ('Content-Length: -1', b'', 400),
    ('Content-Length: abc', b'', 400),
    ('Content-Length: 1', b'\xff', 400),
    ('Content-Length: 2', b'[]', 400),
    (f'Content-Length: {MAX_REQUEST_BYTES + 1}', b'', 413),
    ('Content-Length: 2\r\nContent-Length: 2', b'{}', 400),
    ('Transfer-Encoding: chunked', b'', 400),
])
def test_real_http_rejects_malformed_body_without_hanging(headers, body, expected):
    with HTTPServer(('127.0.0.1', 0), _AgfcRequestHandler) as server:
        worker = threading.Thread(target=server.handle_request, daemon=True)
        worker.start()
        with socket.create_connection(server.server_address, timeout=3) as client:
            client.sendall(f'POST /extract HTTP/1.0\r\n{headers}\r\n\r\n'.encode() + body)
            response = client.recv(4096)
            assert f' {expected} '.encode() in response
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_cli_rejects_invalid_extract_contract_instead_of_reextracting(tmp_path):
    payload = tmp_path / 'invalid.json'
    payload.write_text('[]')
    assert main(['repair', 'mineru', '--source', 'missing.pdf', '--artifact-dir', 'missing',
                 '--output-dir', str(tmp_path / 'out'), '--extract-result', str(payload)]) == 2
    assert not (tmp_path / 'out').exists()


def test_mineru_does_not_reuse_last_same_page_image_and_preserves_assets(tmp_path):
    artifact = tmp_path / 'mineru'
    (artifact / 'images').mkdir(parents=True)
    (artifact / 'images/old.png').write_bytes(b'original')
    original = [{'type': 'image', 'page_idx': 0, 'img_path': 'images/old.png', 'image_caption': ['Retained caption']} for _ in range(2)]
    (artifact / 'content_list.json').write_text(json.dumps(original))
    asset = tmp_path / 'new.png'
    asset.write_bytes(b'new')
    # Global slot 1 used to fall back to the already-used page-0 image.
    extract = {'images': [{'page_idx': 9, 'asset_path': str(asset)}, {'page_idx': 0, 'asset_path': str(asset)}]}
    result = repair_mineru_artifact(source_path=tmp_path / 'source.pdf', artifact_dir=artifact,
                                  output_dir=tmp_path / 'repair', extract_result=extract)
    assert [entry['decision'] for entry in result['replacements']] == ['replace', 'keep_original']
    for key in ['repaired_content_list', 'merged_content_list']:
        path = Path(result['outputs'][key])
        records = json.loads(path.read_text())
        assert (path.parent / records[0]['asset_path']).read_bytes() == b'new'
        assert (path.parent / records[1]['asset_path']).read_bytes() == b'original'
        assert records[0]['img_path'] == records[0]['asset_path']
        assert records[0]['image_caption'] == 'Retained caption'
    assert json.loads((artifact / 'content_list.json').read_text()) == original


def test_mineru_refuses_output_inside_original(tmp_path):
    artifact = tmp_path / 'mineru'
    artifact.mkdir()
    (artifact / 'content_list.json').write_text('[]')
    with pytest.raises(ValueError, match='overlap'):
        repair_mineru_artifact(source_path=tmp_path / 'source.pdf', artifact_dir=artifact,
                              output_dir=artifact, extract_result={'images': []})
    assert list(artifact.iterdir()) == [artifact / 'content_list.json']


@pytest.mark.parametrize('overlay', ['text', 'vector', 'rotation', 'plain'])
def test_object_export_preserves_page_composition(tmp_path, overlay):
    image_bytes = BytesIO()
    Image.new('RGB', (120, 120), '#abcdef').save(image_bytes, format='PNG')
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=200)
        bbox = (30.0, 30.0, 150.0, 150.0)
        xref = page.insert_image(fitz.Rect(bbox), stream=image_bytes.getvalue())
        if overlay == 'text':
            page.insert_text((50, 80), '拱顶 测线', fontname='china-s', fontsize=15)
        elif overlay == 'vector':
            page.draw_line((40, 40), (130, 130), color=(1, 0, 0), width=4)
        elif overlay == 'rotation':
            page.set_rotation(90)
        raw = extract_clean_figure_image(doc, page, figure_bbox=bbox, xref_usage_counts={xref: 1})
        if overlay == 'plain':
            assert raw is not None
            return
        assert raw is None
        rendered = _render_page(page, 144)
        result = export_figure_crops(page_image=rendered, figures=[FigureCandidate(id='figure', bbox=bbox, page_idx=0)],
                                     render_dpi=144, output_dir=tmp_path / 'images', pdf_doc=doc, pdf_page=page,
                                     xref_usage_counts={xref: 1})
        with Image.open(result[0]) as crop:
            expected = rendered.crop(tuple(int(value * 2) for value in bbox))
            assert ImageChops.difference(crop.convert('RGB'), expected.convert('RGB')).getbbox() is None


def test_contract_rejects_incomplete_run_and_does_not_guess_other_assets(tmp_path):
    from agfc.contracts import build_extract_result
    with pytest.raises(FileNotFoundError):
        build_extract_result(tmp_path)
    (tmp_path / 'summary.json').write_text(json.dumps({'pdf': 'source.pdf', 'pages': [{'page_idx': 0}]}))
    page_dir = tmp_path / 'pages/page_000'
    page_dir.mkdir(parents=True)
    (page_dir / 'figures.json').write_text(json.dumps([{'id': 'missing', 'bbox': [0, 0, 10, 10]}]))
    (tmp_path / 'images').mkdir()
    (tmp_path / 'images/page_000_other.png').write_bytes(b'wrong asset')
    with pytest.raises(FileNotFoundError, match='missing'):
        build_extract_result(tmp_path)


def test_demo_contracts_match_shipped_schemas(tmp_path):
    import jsonschema
    from agfc.runtime.demo import run_extract_demo, run_mineru_demo
    from agfc import __version__
    schema_dir = Path(__file__).resolve().parents[1] / 'src/agfc/schemas'
    for name, run_demo in [('extract', run_extract_demo), ('mineru-repair', run_mineru_demo)]:
        result = run_demo(tmp_path / name)
        jsonschema.validate(result, json.loads((schema_dir / f'{name}-result.schema.json').read_text()))
        assert result['engine_version'] == __version__


def test_rotated_pdf_extraction_uses_consistent_page_coordinates(tmp_path):
    from agfc.runtime.demo import create_demo_pdf
    source = create_demo_pdf(tmp_path / 'normal.pdf')
    rotated = tmp_path / 'rotated.pdf'
    with fitz.open(source) as doc:
        doc[0].set_rotation(90)
        doc.save(rotated)
    normal_run = run_pdf(source, output_dir=tmp_path / 'normal_run')
    rotated_run = run_pdf(rotated, output_dir=tmp_path / 'rotated_run')
    for name in ['figures.json', 'atoms.json']:
        assert json.loads((normal_run / 'pages/page_000' / name).read_text()) == json.loads((rotated_run / 'pages/page_000' / name).read_text())
    with fitz.open(rotated) as doc:
        assert doc[0].rotation == 90  # Original file is never rewritten.


def test_dataset_index_matches_authoritative_metadata():
    import csv
    from collections import Counter
    root = Path(__file__).resolve().parents[1] / 'data/private/journalmix_v1'
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = list(csv.DictReader((root / 'page_index.csv').open()))
    families = Counter()
    docs = set()
    for row in rows:
        meta = json.loads((root / 'meta' / f"{row['page_id']}.json").read_text())
        gt = json.loads((root / 'gt' / f"{row['page_id']}.json").read_text())
        assert row['doc_id'] == meta['doc_id']
        assert int(row['page_idx']) == gt['page_idx']
        assert row['figure_family'] == meta['figure_family']
        families[row['figure_family']] += 1
        docs.add(row['doc_id'])
    assert manifest['bucket_counts'] == dict(families)
    assert manifest['doc_count'] == len(docs)


def test_journalmix_missing_source_cannot_fall_back_to_stale_document(tmp_path):
    from agfc.research.journalmix_selected_pages import load_journalmix_selected_page_records
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'gt').mkdir()
    (tmp_path / 'review').mkdir()
    (tmp_path / 'page_index.csv').write_text('page_id,candidate_id,doc_id\njm_0001,c_old,old_doc\n')
    wrong = tmp_path / 'old.pdf'
    wrong.write_bytes(b'wrong')
    (tmp_path / 'review/candidates.csv').write_text(f'candidate_id,doc_id,source_pdf\nc_old,old_doc,{wrong}\n')
    (tmp_path / 'meta/jm_0001.json').write_text(json.dumps({'doc_id': 'new_doc'}))
    (tmp_path / 'gt/jm_0001.json').write_text(json.dumps({'page_idx': 6, 'figures': []}))
    with pytest.raises(FileNotFoundError, match='new_doc'):
        load_journalmix_selected_page_records(tmp_path)
