# AI 기반 도서 추천 챗봇 📚

**종촌고등학교 학생들을 위한 AI 도서 추천 시스템**

자연어 질문을 통해 학교 도서관에 있는 책을 쉽고 빠르게 찾아주는 AI 챗봇입니다. "컴퓨터 공학을 수학I과 엮을 수 있는 책을 추천해줘" 와 같이 복잡하고 창의적인 질문에도 의미를 파악하여 최적의 도서를 추천합니다.

![스크린샷](https://github.com/user-attachments/assets/358b5cc1-3312-4fe6-a9ba-2292f7e71da0)

## ✨ 주요 기능

* **🧠 자연어 이해 (NLU)**: "이런 느낌의 책 찾아줘"와 같은 일상적인 대화로 도서를 검색할 수 있습니다.
* **💡 의미 기반 검색**: 단순 키워드 매칭이 아닌, 문장의 의미와 맥락을 이해하여 관련된 책을 추천합니다.
* **💨 실시간 응답**: 사전 계산된 벡터 임베딩을 통해 빠르고 효율적인 검색을 제공합니다.
* **🖥️ 깔끔한 UI**: 채팅 형식의 미니멀한 인터페이스를 통해 사용자가 서비스에 집중할 수 있도록 돕습니다.
* **📖 상세 정보 제공**: 책 표지, 저자, 줄거리, 분류, 위치 등 상세 정보를 팝업으로 제공하여 탐색을 돕습니다.
* **🔗 학교 도서관 연동**: 검색된 각 도서를 클릭 한 번으로 학교 도서관 검색 결과 페이지로 바로 연결합니다.

## 🛠️ 기술 스택

| 구분 | 기술 |
| :--- | :--- |
| **Backend** | <img src="https://img-shields-io.proxy.start.ig.local/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img-shields-io.proxy.start.ig.local/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"> <img src="https://img-shields-io.proxy.start.ig.local/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white"> |
| **Frontend** | <img src="https://img-shields-io.proxy.start.ig.local/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white"> <img src="https://img-shields-io.proxy.start.ig.local/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white"> <img src="https://img-shields-io.proxy.start.ig.local/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black"> |
| **Data** | <img src="https://img-shields-io.proxy.start.ig.local/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> <img src="https://img-shields-io.proxy.start.ig.local/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"> |
| **Server** | <img src="https://img-shields-io.proxy.start.ig.local/badge/Uvicorn-009688?style=for-the-badge&logo=python&logoColor=white"> |

## ⚙️ 설치 및 실행 방법

### 1. 레포지토리 클론

```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

### 2. Python 가상 환경 설정 및 패키지 설치
이 프로젝트는 다음 라이브러리들을 사용합니다. `requirements.txt` 파일을 통해 한 번에 설치할 수 있습니다.

```bash
# 필요한 패키지 목록
# --------------------
# fastapi: 고성능 웹 프레임워크
# uvicorn[standard]: ASGI 서버
# python-dotenv: .env 파일 관리
# numpy, pandas: 데이터 처리
# google-generativeai: Gemini API 사용
# starlette: FastAPI의 핵심 컴포넌트
# --------------------

# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화 (Windows)
.\venv\Scripts\activate

# 가상 환경 활성화 (macOS/Linux)
source venv/bin/activate

# requirements.txt 파일로 모든 패키지 설치
pip install -r requirements.txt
```

### 3. Google Gemini API 키 설정

1.  [Google AI Studio](https://aistudio.google.com/app/apikey)에서 API 키를 발급받으세요.
2.  프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래와 같이 API 키를 추가합니다.

    ```
    # .env
    GOOGLE_API_KEY="여기에_발급받은_API_키를_붙여넣으세요"
    ```

### 4. 백엔드 서버 실행

```bash
uvicorn app:app --reload
```

### 5. 서비스 접속

웹 브라우저를 열고 `http://127.0.0.1:8000` 주소로 접속하세요.

## 📁 파일 구조

```
.
├── 📂 data/
│   ├── book.csv
│   ├── books_meta.csv
│   └── books_emb.npz
├── 📂 static/
│   ├── style.css
│   └── script.js
├── 📄 app.py
├── 📄 index.html
├── 📄 requirements.txt
├── 📄 .env
└── 📄 README.md
```


일단 여기서 필요한 파일을 다운로드 한 후에 파일 구조에 맞게 배치를 합니다.


## 📜 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.
