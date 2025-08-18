# 📈 고객사 데이터 분석 및 시장 경쟁력 대시보드

> **복잡한 수입 데이터를 직관적인 시각 자료로 변환하여, 영업 담당자가 데이터 기반의 인사이트를 고객에게 효과적으로 전달할 수 있도록 돕는 인터랙티브 웹 대시보드입니다.**

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Badge"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas Badge"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn Badge"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly Badge"/>
</p>

---

## 🎯 The Problem: 데이터의 가치를 설명하기 어려운 영업 환경
기존 영업 담당자들은 고객사(수입사)에게 방대한 Raw 데이터를 제공받아도, 그 안에서 유의미한 가치를 찾아내고 설득력 있게 설명하는 데 어려움을 겪고 있었습니다.

-   **데이터의 복잡성**: TDS(Trade Data Service)와 같은 Raw 데이터는 수많은 행과 열로 이루어져 있어 비전문가가 이해하기 어렵습니다.
-   **직관성 부족**: 숫자와 텍스트로만 가득한 데이터를 보고 고객사의 구매 패턴, 비용 절감 기회, 시장 내 위치를 즉각적으로 파악하기 힘듭니다.
-   **정량적 근거 제시의 어려움**: "저희와 계약하시면 비용을 절감할 수 있습니다"와 같은 추상적인 제안 대신, 구체적인 예상 절감액과 같은 정량적 근거를 제시하기가 까다로웠습니다.

---

## 💡 The Solution: 데이터를 '보여주고' '증명하는' 분석 툴
이 대시보드는 복잡한 데이터 분석 과정을 자동화하고, 그 결과를 누구나 쉽게 이해할 수 있는 인터랙티브 시각화 자료로 제공하여 위 문제를 해결합니다.

-   **고객사 효율 분석**: 계약 전후의 구매 단가 변화를 품목군별로 자동 분석하여, 파트너십을 통해 얻을 수 있는 **예상 절감액을 정량적으로 증명**합니다.
-   **시장 경쟁력 분석**: 고객사의 구매 데이터를 시장 전체 및 주요 경쟁사들과 비교하여, **객관적인 시장 내 위치와 추가적인 비용 절감 기회**를 보여줍니다.
-   **지능형 품목 클러스터링**: `TF-IDF`와 `DBSCAN` 알고리즘을 활용하여, 제각각인 품목명을 자동으로 정제하고 유사한 품목군으로 그룹화하여 분석의 정확도를 높입니다.

---

## ✨ Business Impact
이 툴은 단순한 데이터 시각화를 넘어, 실제 비즈니스 성과에 직접적으로 기여할 수 있는 효과를 창출했습니다.

-   **제안 설득력 강화**: 추상적인 영업 제안을 **데이터 기반의 구체적인 가치 제안**으로 전환했습니다. `"계약 후 약 $253,968의 비용 절감이 예상됩니다"`와 같이 신뢰도 높은 근거를 제시할 수 있게 되었습니다.
-   **고객 이해도 및 신뢰 증진**: 고객에게 자신의 구매 데이터에 대한 깊이 있는 인사이트를 제공함으로써, 단순 판매자를 넘어 **신뢰할 수 있는 데이터 분석 파트너**로서의 입지를 구축했습니다.
-   **새로운 영업 기회 발굴**: 시장 내 경쟁사 및 공급망 분석을 통해 고객이 미처 인지하지 못했던 **대안 소싱 옵션이나 추가적인 비용 절감 포인트**를 선제적으로 제안할 수 있게 되었습니다.
-   **내부 업무 효율성 증대**: 수동으로 몇 시간씩 걸리던 데이터 분석 및 보고서 작성 업무를 자동화하여, 영업 담당자가 **핵심적인 고객 관계 구축 활동**에 더 집중할 수 있도록 만들었습니다.

---

## 🚀 로컬에서 실행하기

### **Prerequisites**
-   Python 3.8+

### **Installation & Setup**
1.  **저장소 복제 (Clone the repository)**:
    ```bash
    git clone [저장소 URL]
    cd [프로젝트 폴더]
    ```

2.  **필요한 라이브러리 설치**:
    *프로젝트 폴더에 `requirements.txt` 파일을 생성하고 아래 내용을 추가한 후, 터미널에서 설치 명령어를 실행하세요.*

    **requirements.txt**:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    streamlit-option-menu
    statsmodels
    ```

    **터미널**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Streamlit 앱 실행**:
    ```bash
    streamlit run your_app_name.py
    ```

4. **streamlit cloud 앱 실행**:
   demo-for-test.streamlit.app
   
