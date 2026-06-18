import os
import json

# Get path relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, "../data/train.jsonl"))
DEFAULT_OUTPUT_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, "../data/train_chat.jsonl"))

def convert_to_chat_format(input_file, output_file):
    """
    기존 instruction/input/output 형식을 chat 형식으로 변환
    
    Args:
        input_file: 원본 JSONL 파일 경로 (예: 'train.jsonl')
        output_file: 변환된 파일 저장 경로 (예: 'train_chat.jsonl')
    """
    print(f"Converting {input_file} to chat format...")
    
    converted_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # 원본 데이터 로드
                sample = json.loads(line.strip())
                
                # Chat 형식으로 변환
                chat_sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": sample["instruction"]
                        },
                        {
                            "role": "user",
                            "content": json.dumps(sample["input"], ensure_ascii=False)
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(sample["output"], ensure_ascii=False)
                        }
                    ]
                }
                
                # 한 줄씩 저장
                outfile.write(json.dumps(chat_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                # 진행상황 표시 (1000개마다)
                if converted_count % 1000 == 0:
                    print(f"  Converted {converted_count} samples...")
                    
            except Exception as e:
                print(f"  ⚠️  Error at line {line_num}: {e}")
                continue
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total converted: {converted_count} samples")
    print(f"   Saved to: {output_file}")
    
    # 샘플 미리보기
    print(f"\n📋 Sample preview (first item):")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_sample = json.loads(f.readline())
        print(json.dumps(first_sample, indent=2, ensure_ascii=False)[:500] + "...")

if __name__ == "__main__":
    # 사용 예시
    INPUT_FILE = DEFAULT_INPUT_FILE  # 기존 파일
    OUTPUT_FILE = DEFAULT_OUTPUT_FILE  # 변환된 파일
    
    convert_to_chat_format(INPUT_FILE, OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("🎯 Next step: 파인튜닝 코드에서 파일명만 변경하세요!")
    print('   CONFIG["data_path"] = "../data/train_chat.jsonl"')
    print("="*60)