import json
import re

def fix_jsonl_format(input_file, output_file):
    """
    잘못된 JSONL 형식을 올바른 형식으로 수정합니다.
    각 JSON 객체를 줄바꿈으로 구분합니다.
    """
    try:
        # 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"파일 {input_file}이 비어있습니다.")
            return
        
        # JSON 객체들을 분리하기 위해 정규표현식 사용
        # }{"를 }\n{"로 바꿔서 줄바꿈 추가
        fixed_content = re.sub(r'}\s*{', '}\n{', content)
        
        # 수정된 내용을 새 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # 유효성 검사 - 각 라인이 올바른 JSON인지 확인
        valid_lines = 0
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        valid_lines += 1
                    except json.JSONDecodeError as e:
                        print(f"라인 {line_num}에서 JSON 오류: {e}")
                        print(f"문제가 있는 라인: {line[:100]}...")
        
        print(f"✅ JSONL 형식 수정 완료!")
        print(f"   입력 파일: {input_file}")
        print(f"   출력 파일: {output_file}")
        print(f"   유효한 JSON 라인 수: {valid_lines}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    # 원본 파일을 수정
    fix_jsonl_format('gri_data_fixed.jsonl', 'gri_data_fixed_corrected.jsonl') 