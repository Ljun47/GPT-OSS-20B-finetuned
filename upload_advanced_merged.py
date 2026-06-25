from huggingface_hub import HfApi

api = HfApi()

# 1. 업로드할 로컬의 심화 프로젝트 병합 완료 모델 경로
LOCAL_FOLDER = "/Users/jun/Downloads/취업/심화 프로젝트/gpt-oss/gpt-oss-20b-trading-merged"

# 2. 허깅페이스에 생성할 모델 레포지토리 이름 (계정명/원하는이름)
REPO_ID = "jun47/gpt-oss-20b-trading-merged"

print("1. 허깅페이스에 20B 병합 모델 전용 레포지토리 생성 중...")
api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    private=False  # 외부 시연 및 포트폴리오용이므로 Public 설정
)

print(f"2. {LOCAL_FOLDER} 폴더 전체를 {REPO_ID}로 업로드 시작 (용량이 크므로 수 분 소요될 수 있음)...")
api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model"
)
print(f"🎉 심화 프로젝트 20B 병합 완료 모델 업로드 성공!")
print(f"👉 확인 주소: https://huggingface.co/{REPO_ID}")
