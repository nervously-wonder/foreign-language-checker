"""
외래어 표기법 검사기
Flask 백엔드: 후보 추출 → 국립국어원 API
"""

import os
import re
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from kiwipiepy import Kiwi

logging.basicConfig(level=logging.INFO)
load_dotenv()

app = Flask(__name__)

KORNORMS_API_KEY = os.getenv("KORNORMS_API_KEY")
KORNORMS_API_URL = "https://korean.go.kr/kornorms/exampleReqList.do"

kiwi = Kiwi()

# ── 흔한 한국어 단어 (고유명사 후보에서 제외) ────────────────────
_COMMON_KOREAN = {
    "우리", "자신", "여성", "남성", "사람", "누구", "모두", "하나",
    "작품", "예술", "공간", "사진", "제목", "소재", "감각", "개념",
    "대상", "경험", "형태", "방식", "지식", "균형", "의미", "관계",
    "해방", "행위", "요소", "초점", "기반", "경계", "범주", "재료",
    "연상", "생각", "접촉", "표면", "정서", "촉각", "시각", "미학",
    "섬유", "장식", "변형", "일상", "전시", "운동", "황동", "관객",
    "부담", "온도", "혼란", "감탄", "지면", "지위", "열쇠", "온몸",
    "쾌감", "고통", "직조", "보풀", "주렴", "감상", "체화", "함의",
    "이해", "조율", "형식", "정수", "사실", "수용", "고유", "제안",
    "포함", "토대", "토론", "투자", "태도", "파악", "파괴", "판단",
    "폭력", "피해", "학교", "학생", "탄생", "태양", "특히", "통해",
    "코너", "코드", "아티스트", "가능", "가족", "감독", "감정",
    "결과", "결국", "관점", "교육", "구조", "기술", "기억", "기준",
    "기타", "노력", "도구", "도시", "독자", "동시", "목적", "문제",
    "문화", "물질", "미래", "방법", "본질", "부분", "분야", "불가",
    "비롯", "사건", "사회", "상태", "색채", "성격", "세계", "소설",
    "속도", "수준", "시대", "시작", "실제", "영향", "예측", "완성",
    "위치", "이후", "인간", "인물", "자체", "장소", "전체", "정보",
    "정치", "조건", "존재", "주요", "중요", "증가", "지역", "차이",
    "최초", "추구", "표현", "필요", "현대", "현실", "확인", "환경",
    "활동", "효과", "패턴", "미학적", "무제", "평론", "머리카락", "잉태", "포착",
    "그리고", "하지만", "그러나", "따라서", "또한", "이것", "저것", "그것",
    "평론가", "연구자", "학자", "큐레이터", "편집자", "번역가", "기획자",
    "조우", "대화", "충돌", "탈출", "귀환", "출발", "도착", "접근",
}


def _has_foreign_pattern(word):
    foreign_end = set('크트프스즈드그브흐')
    aspirated = set('카타파키티피쿠투푸코토포케테페캐태패')
    foreign_syllables = set('핸폰브젝션웨워랜렌런벨젤맨톤넷벳펀')
    if word[-1] in foreign_end:
        return True
    if any(ch in aspirated for ch in word):
        return True
    if '르' in word or '슈' in word or '츠' in word or '오브' in word:
        return True
    if sum(1 for ch in word if ch in foreign_syllables) >= 2:
        return True
    return False


def _has_strong_foreign_pattern(word):
    strong_patterns = ['날레', '나레', '스터', '슈타', '비엔', '갈레', '르세']
    if any(p in word for p in strong_patterns):
        return True
    foreign_end = set('크트프스즈드그브흐')
    aspirated = set('카타파키티피쿠투푸코토포케테페캐태패')
    score = 0
    if word[-1] in foreign_end:
        score += 2
    score += sum(1 for ch in word if ch in aspirated)
    if '르' in word or '슈' in word or '츠' in word:
        score += 1
    return score >= 3


def _trim_prefix(name):
    prefix_words = {
        "아티스트", "작가", "감독", "배우", "교수", "박사", "선수",
        "대통령", "총리", "장관", "화가", "작곡가", "철학자", "소설가",
        "시인", "건축가", "디자이너", "프로듀서", "기자", "평론가",
        "연구자", "학자", "사진가", "큐레이터", "편집자", "번역가",
        "성악가", "지휘자", "연출가", "작사가", "작가", "기획자",
    }
    parts = name.split()
    while len(parts) > 1 and parts[0] in prefix_words:
        parts.pop(0)
    return " ".join(parts)


def _is_likely_korean_phrase(text):
    words = text.split()
    if len(words) <= 2:
        return False
    common_count = sum(1 for w in words if w in _COMMON_KOREAN or len(w) == 1)
    return common_count >= len(words) // 2


def extract_candidates(text):
    """텍스트에서 외래어 후보를 추출한다. 원어 철자 맵도 함께 반환."""
    candidates = set()
    bracket_candidates = set()  # 괄호 패턴에서 추출한 신뢰도 높은 후보
    original_map = {}  # 한국어 → 원어 철자

    # 괄호 패턴 — "홍길동(Hong Gildong)" 형태
    paren_pattern = re.compile(r'([가-힣]+(?:\s[가-힣]+){0,3})\s*\(([A-Za-z][\w\s\.\-\']+)\)')
    for match in paren_pattern.finditer(text):
        korean_name = _trim_prefix(match.group(1).strip())
        orig_spelling = match.group(2).strip()
        words = korean_name.split()
        foreign_words = []
        for w in reversed(words):
            if w in _COMMON_KOREAN or (not _has_foreign_pattern(w) and len(w) <= 2):
                break
            foreign_words.insert(0, w)
        if foreign_words:
            orig_words = orig_spelling.split()
            n_orig = len(orig_words)
            for k, w in enumerate(reversed(foreign_words)):
                if len(w) >= 2:
                    actual_w = foreign_words[len(foreign_words) - 1 - k]
                    candidates.add(actual_w)
                    bracket_candidates.add(actual_w)
                    if actual_w not in original_map:
                        # 뒤에서부터 위치 대응
                        original_map[actual_w] = orig_words[n_orig - 1 - k] if k < n_orig else orig_spelling
        else:
            continue

    # kiwipiepy 형태소 분석 — NNP 연속 병합
    result = kiwi.analyze(text)
    tokens = result[0][0]

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.tag == "NNP":
            merged_end = token.start + token.len
            parts = [token.form]
            j = i + 1
            while j < len(tokens):
                next_t = tokens[j]
                gap = next_t.start - merged_end
                if next_t.tag == "NNP" and 0 <= gap <= 1:
                    if gap == 1:
                        parts.append(" ")
                    parts.append(next_t.form)
                    merged_end = next_t.start + next_t.len
                    j += 1
                else:
                    break
            merged = "".join(parts)
            i = j
            if " " in merged:
                # 전체 복합 형태 우선 추가
                all_korean = all('가' <= ch <= '힣' or ch == ' ' for ch in merged)
                if all_korean and not _is_likely_korean_phrase(merged):
                    candidates.add(merged)
            else:
                clean = merged.replace(" ", "")
                if len(clean) >= 2 and all('가' <= ch <= '힣' for ch in clean):
                    if not _is_likely_korean_phrase(merged):
                        candidates.add(merged)
        else:
            i += 1

    # NNG 중 외래어 패턴 (강한 패턴만)
    for token in tokens:
        if token.tag == "NNG" and len(token.form) >= 2:
            if all('가' <= ch <= '힣' for ch in token.form):
                if _has_strong_foreign_pattern(token.form) and token.form not in _COMMON_KOREAN:
                    candidates.add(token.form)

    # 조사 제거 후 외래어 패턴
    josa_pattern = re.compile(
        r'(은|는|이|가|을|를|에|의|도|로|으로|와|과|에서|에게|부터|까지|처럼|라는|이란|에는|에서는)$'
    )
    for raw_word in re.findall(r'[가-힣]{3,}', text):
        stripped = josa_pattern.sub('', raw_word)
        if len(stripped) >= 3 and stripped != raw_word and _has_foreign_pattern(stripped):
            candidates.add(stripped)
        if len(stripped) >= 3 and stripped not in _COMMON_KOREAN:
            if _has_strong_foreign_pattern(stripped):
                candidates.add(stripped)

    # 필터링
    filtered = []
    for c in sorted(candidates):
        clean = c.replace(" ", "")
        if len(clean) <= 1 or clean in _COMMON_KOREAN:
            continue
        filtered.append(c)

    # 서브셋 중복 제거
    to_remove = set()
    for i, a in enumerate(filtered):
        a_clean = a.replace(" ", "")
        for j, b in enumerate(filtered):
            if i == j or i in to_remove or j in to_remove:
                continue
            b_clean = b.replace(" ", "")
            if b_clean in a_clean and len(a_clean) > len(b_clean):
                a_has_korean = any(w in _COMMON_KOREAN for w in a.split())
                if a_has_korean:
                    to_remove.add(i)
                elif b in bracket_candidates and a not in bracket_candidates:
                    # 괄호 패턴 결과(b)는 kiwipiepy 결과(a)보다 우선 보존
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    deduped = [f for i, f in enumerate(filtered) if i not in to_remove]

    # 다중어 후보의 마지막 단어가 다른 후보의 접두어인 경우 분리
    # 예: '터너 컨템포러' + '컨템포러리' → '터너', '컨템포러리'
    to_add = []
    to_drop = set()
    for a in deduped:
        if ' ' not in a:
            continue
        parts = a.split()
        last = parts[-1]
        for b in deduped:
            if b != a and b.startswith(last) and len(b) > len(last):
                prefix = ' '.join(parts[:-1])
                if len(prefix.replace(' ', '')) >= 2:
                    to_add.append(prefix)
                to_drop.add(a)
                break

    result = [c for c in deduped if c not in to_drop] + to_add
    return result, original_map


# ── 국립국어원 API ────────────────────────────────────────────────

def search_kornorms(keyword, search_type="equal"):
    if not KORNORMS_API_KEY:
        return None
    params = {
        "serviceKey": KORNORMS_API_KEY,
        "pageNo": 1,
        "numOfRows": 10,
        "langType": "0003",
        "resultType": "json",
        "searchEquals": search_type,
        "searchKeyword": keyword,
    }
    try:
        resp = requests.get(KORNORMS_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def parse_items(data):
    if not data:
        return []
    try:
        return data.get("response", {}).get("items") or []
    except (AttributeError, TypeError):
        return []


_CONFUSABLE_PAIRS = [
    ('샵', '숍'), ('숍', '샵'), ('칼', '갈'), ('갈', '칼'),
    ('셋', '세'), ('세', '셋'), ('쉬', '슈'), ('슈', '쉬'),
    ('씨', '시'), ('시', '씨'), ('빠', '파'), ('파', '빠'),
    ('까', '카'), ('카', '까'), ('따', '타'), ('타', '따'),
    ('렌', '런'), ('런', '렌'), ('벨', '밸'), ('밸', '벨'),
    ('워', '위'), ('위', '워'), ('왜', '웨'), ('웨', '왜'),
    ('애', '에'), ('에', '애'), ('오', '어'), ('어', '오'),
]


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
        prev_row = curr_row
    return prev_row[-1]


def _generate_variants(word):
    variants = set()
    for old, new in _CONFUSABLE_PAIRS:
        if old in word:
            variants.add(word.replace(old, new, 1))
    return sorted(variants, key=lambda v: levenshtein(word, v))[:5]


def is_similar(w1, w2):
    if w1 == w2 or w1 in w2 or w2 in w1:
        return True
    if levenshtein(w1, w2) <= 2:
        return True
    if len(w1) >= 2 and len(w2) >= 2 and w1[:2] == w2[:2]:
        return True
    if abs(len(w1) - len(w2)) <= 2:
        common = sum(1 for ch in w1 if ch in w2)
        if common / max(len(w1), len(w2)) >= 0.4:
            return True
    return False


def is_chinese(item):
    lang = item.get("lang_nm", "") or ""
    guk = item.get("guk_nm", "") or ""
    if "중국어" in lang or "일본어" in lang:
        return True
    if "중국" in guk or "일본" in guk:
        return True
    if re.search(r'[\u4e00-\u9fff]', item.get("srclang_mark", "")):
        return True
    return False


def find_typo_candidates(word, custom_dict, max_distance=2):
    candidates = []
    for dict_key, dict_val in custom_dict.items():
        dist = levenshtein(word, dict_key)
        if 0 < dist <= max_distance:
            candidates.append({"word": dict_val if dict_val != dict_key else dict_key,
                                "distance": dist, "source": "자체 사전"})
    return sorted(candidates, key=lambda x: x["distance"])[:3]


# ── 단어 검사 ─────────────────────────────────────────────────────

def check_word(word, custom_dict=None, original=''):
    """단어 하나를 검사해서 결과를 반환한다."""
    clean = word.replace(" ", "")

    # 자체 사전
    if custom_dict:
        if clean in custom_dict:
            val = custom_dict[clean]
            if val == clean:
                return {"word": word, "status": "custom_correct", "source": "자체 사전"}
            else:
                return {"word": word, "status": "custom_fix", "correction": val, "source": "자체 사전"}
        typo_hits = find_typo_candidates(clean, custom_dict)
        if typo_hits:
            return {"word": word, "status": "typo_candidate", "similar": typo_hits}

    def make_correct(item):
        return {
            "word": word, "status": "correct",
            "korean": item.get("korean_mark", "").strip(),
            "original": item.get("srclang_mark", ""),
            "country": item.get("guk_nm", ""),
            "language": item.get("lang_nm", ""),
            "category": item.get("foreign_gubun", ""),
        }

    # 원어 병기가 있으면 원어로 먼저 검색
    if original:
        time.sleep(0.15)
        data = search_kornorms(original, "equal")
        items = parse_items(data)
        for item in items:
            if is_chinese(item):
                continue
            src = item.get("srclang_mark", "").strip()
            korean = item.get("korean_mark", "").strip()
            if src.lower() == original.lower():
                if korean == clean:
                    return make_correct(item)
                else:
                    # 원어는 맞지만 한국어 표기가 다름
                    return {
                        "word": word, "status": "check",
                        "suggestions": [{
                            "korean": korean,
                            "original": src,
                            "country": item.get("guk_nm", ""),
                            "language": item.get("lang_nm", ""),
                            "category": item.get("foreign_gubun", ""),
                            "distance": levenshtein(clean, korean),
                        }],
                    }

    # 1차: 한국어 정확 일치
    data = search_kornorms(clean, "equal")
    items = parse_items(data)
    if items:
        for item in items:
            korean = item.get("korean_mark", "").strip()
            if korean == clean and not is_chinese(item):
                return make_correct(item)

    time.sleep(0.15)

    # 2차: 부분 일치
    data = search_kornorms(clean, "like")
    items = parse_items(data)
    if items:
        relevant = []
        for item in items:
            korean = item.get("korean_mark", "").strip()
            if not korean or is_chinese(item):
                continue
            if korean == clean:
                return make_correct(item)
            if is_similar(clean, korean):
                relevant.append({
                    "korean": korean,
                    "original": item.get("srclang_mark", ""),
                    "country": item.get("guk_nm", ""),
                    "language": item.get("lang_nm", ""),
                    "category": item.get("foreign_gubun", ""),
                    "distance": levenshtein(clean, korean),
                })
        if relevant:
            relevant.sort(key=lambda x: x.get("distance", 99))
            return {"word": word, "status": "check", "suggestions": relevant[:3]}

    # 3차: 변형 패턴 검색
    variants = _generate_variants(clean)
    for idx, variant in enumerate(variants):
        if idx >= 3:
            break
        time.sleep(0.15)
        data = search_kornorms(variant, "equal")
        items = parse_items(data)
        if items:
            for item in items:
                korean = item.get("korean_mark", "").strip()
                guk = item.get("guk_nm", "") or ""
                if korean == variant and not is_chinese(item):
                    return {
                        "word": word, "status": "check",
                        "suggestions": [{
                            "korean": korean,
                            "original": item.get("srclang_mark", ""),
                            "country": guk,
                            "language": item.get("lang_nm", ""),
                            "category": item.get("foreign_gubun", ""),
                            "distance": levenshtein(clean, korean),
                        }],
                    }

    if not KORNORMS_API_KEY:
        return {"word": word, "status": "no_api_key"}

    return {"word": word, "status": "not_found", "original": original}


# ── Flask 라우트 ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def api_extract():
    text = request.json.get("text", "")
    if not text.strip():
        return jsonify({"candidates": []})
    candidates, original_map = extract_candidates(text)
    result = [{"word": w, "original": original_map.get(w, "")} for w in candidates]
    return jsonify({"candidates": result})


@app.route("/api/check", methods=["POST"])
def api_check():
    try:
        words_raw = request.json.get("words", [])
        custom_dict = request.json.get("custom_dict", {}) or None

        def parse_entry(entry):
            if isinstance(entry, dict):
                return entry.get("word", ""), entry.get("original", "")
            return entry, ""

        seen = set()
        unique_entries = []
        for entry in words_raw:
            w, orig = parse_entry(entry)
            clean = w.replace(" ", "")
            if clean not in seen:
                seen.add(clean)
                unique_entries.append((w, orig))

        results_map = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_word = {
                executor.submit(check_word, w, custom_dict, orig): w
                for w, orig in unique_entries
            }
            for future in as_completed(future_to_word):
                word = future_to_word[future]
                try:
                    results_map[word] = future.result()
                except Exception as e:
                    logging.error(f"check_word 오류 ({word}): {e}")
                    results_map[word] = {"word": word, "status": "error"}

        results = [results_map[w] for w, _ in unique_entries]
        return jsonify({"results": results})

    except Exception as e:
        logging.error(f"api_check 오류: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
