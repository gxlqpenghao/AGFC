from __future__ import annotations

import re

from agfc.models import BBox, PageAtom


FIGURE_CAPTION_RE = re.compile(r"^\s*(?:fig(?:ure)?\.?|图|圖)\s*[\divxlcdm]+[.:：、]?", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*(?:table|tab\.|表)\s*[\divxlcdm]+(?:[.:：、]|\b)", re.IGNORECASE)


def collect_nonraster_annotation_atoms(
    atoms: list[PageAtom],
    *,
    content_bbox: BBox,
) -> list[PageAtom]:
    return [atom for atom in atoms if is_nonraster_annotation_atom(atom, content_bbox=content_bbox)]


def is_nonraster_annotation_atom(
    atom: PageAtom,
    *,
    content_bbox: BBox,
) -> bool:
    if atom.kind != "text_block":
        return False
    raw_text = (atom.text or "").strip()
    if not raw_text:
        return False
    if not _is_near_annotation_extent(atom.bbox, content_bbox):
        return False
    if _looks_like_excluded_visual_text(raw_text):
        return False
    if _looks_like_cjk_body_or_form_text(raw_text):
        return False
    return _looks_like_visual_annotation_text(raw_text)


def looks_like_table_cell_text(raw_text: str) -> bool:
    text = " ".join(raw_text.strip().split())
    if not text:
        return False
    if _looks_like_axis_or_legend_annotation(raw_text):
        return False
    if _starts_like_table_context(text):
        return True
    if _looks_like_cjk_table_cell_text(text):
        return True
    if re.match(
        r"^(?:User|Ground Truth|Model|Format|GPT|GPT-?4|GPT-?4V|LLaVA|ViP|InstructBLIP|Qwen|Kosmos|Shikra)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:GPT-?4V|LLaVA|ViP|InstructBLIP|Qwen|Kosmos|Shikra|InternLM|GeminiProVision|DeepSeek|Claude)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    if len(tokens) < 6:
        return False
    numeric_count = sum(1 for token in tokens if token[0].isdigit())
    return numeric_count / len(tokens) >= 0.3


def _looks_like_excluded_visual_text(raw_text: str) -> bool:
    text = " ".join(raw_text.strip().split())
    if FIGURE_CAPTION_RE.match(text) or TABLE_CAPTION_RE.match(text):
        return True
    if _looks_like_page_number_text(text):
        return True
    if (
        _looks_like_document_running_header_text(text)
        or _looks_like_equation_block_text(raw_text)
        or _looks_like_formula_or_body_math_text(raw_text)
        or _looks_like_section_heading_text(text)
    ):
        return True
    if _looks_like_axis_or_legend_annotation(raw_text):
        return False
    if _looks_like_cjk_body_or_form_text(text):
        return True
    if re.match(r"^\s*(?:[A-Z]\.|[IVXLCDM]+\.)\s+[A-Z][A-Za-z -]{3,}", text):
        return True
    if looks_like_table_cell_text(raw_text):
        return True
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    return len(tokens) >= 8


def _looks_like_document_running_header_text(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    return re.fullmatch(r"[A-Z]\.\s+[A-Z][A-Za-z-]+ et al\.?", normalized) is not None


def _looks_like_equation_block_text(text: str) -> bool:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return False
    if re.search(r"\(\d{1,3}\)\s*$", normalized) is None:
        return False
    return re.search(r"[=+−×/{}|]", normalized) is not None


def _looks_like_formula_or_body_math_text(text: str) -> bool:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return False
    if _looks_like_axis_or_legend_annotation(normalized):
        return False
    math_signal = re.search(r"[=≜∈∑∏√≤≥≈∞{}\[\]^]|(?:\.\s*){2,}|[θκπλμξ]", normalized, re.IGNORECASE)
    if math_signal is None:
        return False
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", normalized)
    return len(tokens) >= 6 or len(normalized) >= 42


def _looks_like_visual_annotation_text(raw_text: str) -> bool:
    text = " ".join(raw_text.strip().split())
    if _looks_like_repeated_ellipsis_annotation(text):
        return True
    if _looks_like_axis_or_legend_annotation(raw_text):
        return True
    if re.fullmatch(r"\(?[a-zA-Z]\)?", text):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return True
    if _looks_like_axis_variable_annotation(text):
        return True
    if _looks_like_short_cjk_diagram_label(text):
        return True
    if len(text) > 90:
        return False
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    if not tokens:
        return False
    if len(tokens) <= 6:
        return True
    return False


def _looks_like_repeated_ellipsis_annotation(text: str) -> bool:
    if not text or len(text) > 36:
        return False
    return re.fullmatch(r"(?:\.{2,}|…)(?:\s+(?:\.{2,}|…))*", text) is not None


def _looks_like_axis_variable_annotation(text: str) -> bool:
    if len(text) > 48:
        return False
    return re.search(r"[αβγδεθκλμρσφψωξ]", text, re.IGNORECASE) is not None or (
        len(re.findall(r"\b[xyz](?:[A-Z]{1,3})?\b", text)) >= 2
    )


def _looks_like_short_cjk_diagram_label(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not re.fullmatch(r"[\u3400-\u9fff]{2,8}", compact):
        return False
    return re.search(
        r"(?:拱顶|拱腰|墙角|台阶|测线|掌子面|界面|透镜|空洞|断面|"
        r"^[左右上下前后内外中](?:拱|墙|角|腰|顶|底|线|侧|端|部|面|层|段))",
        compact,
    ) is not None


def _starts_like_table_context(text: str) -> bool:
    return (
        re.match(
            r"^(?:Dataset|Example File|Task|Reference|Question Statement|Statement:|Model|Format|"
            r"User|Ground Truth|GPT|GPT-?4|GPT-?4V|LLaVA|ViP|InstructBLIP|Qwen|Kosmos|Shikra)\b",
            text,
            re.IGNORECASE,
        )
        is not None
    )


def _looks_like_cjk_table_cell_text(text: str) -> bool:
    if not _contains_cjk(text):
        return False
    cjk_count = _cjk_char_count(text)
    compact = re.sub(r"\s+", "", text)
    if _looks_like_cjk_form_field_label(compact):
        return True
    if re.match(r"^[（(]?\d{1,3}[）).、]\s*[\u3400-\u9fff“\"《]", text) and cjk_count >= 10:
        return True
    if re.match(r"^[一二三四五六七八九十百]+[、.．]\s*[\u3400-\u9fff]", text) and cjk_count >= 8:
        return True
    if re.search(r"(?:项目|课题|计划|基金|工程|单位|日期|编号|类别|来源|行业|专业|期限|关键词)", compact):
        return cjk_count >= 4 and (":" in text or "：" in text or bool(re.match(r"^[（(]?\d", text)))
    return False


def _looks_like_cjk_form_field_label(compact_text: str) -> bool:
    if not compact_text or not _contains_cjk(compact_text):
        return False
    if not (2 <= _cjk_char_count(compact_text) <= 14):
        return False
    return re.search(
        r"(?:名称|编号|日期|单位|行业|专业|来源|类别|期限|关键词|主题词|项目|姓名|奖项|密级|备注|是否公布)$",
        compact_text,
    ) is not None


def _looks_like_cjk_body_or_form_text(text: str) -> bool:
    if not _contains_cjk(text):
        return False
    compact = re.sub(r"\s+", "", text)
    cjk_count = _cjk_char_count(compact)
    if _looks_like_cjk_form_field_label(compact):
        return True
    if _looks_like_cjk_table_cell_text(text):
        return True
    if cjk_count >= 18 and re.search(r"[，。；：:“”《》（）]", compact):
        return True
    if cjk_count >= 24 and len([line for line in text.splitlines() if line.strip()]) <= 2:
        return True
    if _looks_like_axis_or_legend_annotation(text):
        return False
    return False


def _looks_like_page_number_text(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    return re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", normalized) is not None


def _contains_cjk(text: str) -> bool:
    return re.search(r"[\u3400-\u9fff]", text) is not None


def _cjk_char_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", text))


def _looks_like_axis_or_legend_annotation(raw_text: str) -> bool:
    text = " ".join(raw_text.strip().split())
    if not text or len(text) > 180:
        return False
    if _looks_like_page_number_text(text):
        return False
    if _starts_like_table_context(text):
        return False
    if _looks_like_axis_variable_annotation(text):
        return True
    if _looks_like_multiline_legend_list(raw_text):
        return True
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    if not tokens:
        return False
    numeric_count = sum(1 for token in tokens if token[0].isdigit())
    if numeric_count >= 2 and len(tokens) <= 18 and numeric_count / len(tokens) >= 0.35:
        return True
    if len(lines) >= 2 and len(tokens) <= 16 and numeric_count >= 2:
        return True
    if _looks_like_unit_annotation(text, tokens):
        return True
    if 2 <= len(lines) <= 6 and len(tokens) <= 18 and all(len(line) <= 48 for line in lines):
        return True
    return False


def _looks_like_unit_annotation(text: str, tokens: list[str]) -> bool:
    if len(text) > 90 or len(tokens) > 14:
        return False
    return re.search(
        r"(?:\[[^\]]{1,28}\]|\((?=[^)]*[A-Za-zμµ°/%])[A-Za-z0-9μµ°/%.\s-]{1,18}\)|\d+(?:\.\d+)?\s*%|/[A-Za-z])",
        text,
    ) is not None


def _looks_like_section_heading_text(text: str) -> bool:
    return (
        re.match(r"^(?:[A-Z]\.\d+|\d+(?:\.\d+)+)\.?\s+[A-Z]", text) is not None
        or re.match(r"^[A-Z]\.\s+[A-Z][A-Za-z -]{3,}", text) is not None
    )


def _looks_like_multiline_legend_list(raw_text: str) -> bool:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not (3 <= len(lines) <= 8):
        return False
    if any(len(line) > 42 for line in lines):
        return False
    text = " ".join(lines)
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    if not (3 <= len(tokens) <= 24):
        return False
    numeric_count = sum(1 for token in tokens if token[0].isdigit())
    return numeric_count / len(tokens) <= 0.4


def _is_near_annotation_extent(
    text_bbox: BBox,
    content_bbox: BBox,
) -> bool:
    if _bbox_overlap_coverage(text_bbox, content_bbox) > 0.0:
        return True
    cx0, cy0, cx1, cy1 = content_bbox
    tx0, ty0, tx1, ty1 = text_bbox
    content_width = max(1.0, cx1 - cx0)
    content_height = max(1.0, cy1 - cy0)
    max_gap = max(16.0, min(36.0, min(content_width, content_height) * 0.16))

    horizontal_overlap = max(0.0, min(cx1, tx1) - max(cx0, tx0))
    text_width = max(1.0, tx1 - tx0)
    if horizontal_overlap / min(content_width, text_width) >= 0.35:
        if 0.0 <= cy0 - ty1 <= max_gap:
            return True
        if 0.0 <= ty0 - cy1 <= max_gap:
            return True

    vertical_overlap = max(0.0, min(cy1, ty1) - max(cy0, ty0))
    text_height = max(1.0, ty1 - ty0)
    if vertical_overlap / min(content_height, text_height) >= 0.35:
        if 0.0 <= cx0 - tx1 <= max_gap:
            return True
        if 0.0 <= tx0 - cx1 <= max_gap:
            return True
    return False


def _bbox_overlap_coverage(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0
